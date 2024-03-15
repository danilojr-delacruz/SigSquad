import numpy as np
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
from scipy.signal import butter, lfilter
import signatory
import torch

RESIDUAL_PAIRS = {'LP': [('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1')], 
                  'RP': [('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2')], 
                  'LT': [('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1')],
                  'RT': [('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2')],
                  # don't include the middle electrodes for now
                  #'C': [('Fz', 'Cz'), ('Cz', 'Pz')],
                  }

TARGETS = [
    "seizure_vote",	"lpd_vote", "gpd_vote",
    "lrda_vote", "grda_vote", "other_vote"
    ]

def modify_metadata(metadata):
    """Reduce the metadata to one data point per eeg_id (the one in the middle - median offset).
       Make the toal vote distribution across the reconding be the target.
       We are assuming that even though we have multiple sub-recordings, the true target value does not change.
    """
    # note that other public notebooks calculate the offset differently, but I am not convinced it makes sense
    metadata_grouped = metadata.groupby("eeg_id").agg(
        spectrogram_id     = pd.NamedAgg("spectrogram_id", "first"),
        eeg_offset_seconds = pd.NamedAgg("eeg_label_offset_seconds", "median"),
        spec_offset_seconds = pd.NamedAgg("spectrogram_label_offset_seconds", "median"),
        patient_id         = pd.NamedAgg("patient_id", "first"),
        target             = pd.NamedAgg("expert_consensus", "first")
        )
    total_votes = metadata.groupby("eeg_id")[TARGETS].agg("sum")
    total_votes = total_votes.div(total_votes.sum(axis=1), axis=0)
    for vote_label in TARGETS:
        metadata_grouped[vote_label] = total_votes[vote_label]

    return metadata_grouped.reset_index()

def rescale(ts, scaler_type):
   """Rescale the time series using the given type.
   """
   if scaler_type == "minmax":
      scaler = TimeSeriesScalerMinMax()
      ts = scaler.fit_transform(ts)
   elif scaler_type == "meanvar":
      scaler = TimeSeriesScalerMeanVariance(std=0.2)
      ts = scaler.fit_transform(ts)
   elif scaler_type == "constant":
      # since we clipped the data to -300, 300, this will force the data to have range 1
      # this seems to help create well-behaved signatures (e.g. they have factorial decay)
      # we might need to tune this constant if it doesn't work well
      ts = ts / 600
   return ts

def transform_residuals(residuals, scaler_type):
    # clip roughly 3 standard deviations and above
    residuals = np.clip(residuals, -300, 300)
    residuals = rescale(residuals.values.reshape(1,-1,1), scaler_type).reshape(-1)
    return residuals


def get_residuals(eeg, scaler_type):
    """Doctors look at the difference between two neighboring channels.
       Calculate the residuals for each channel pair.
       Group by brain region."""
    brain_regions = []
    for region, pair in RESIDUAL_PAIRS.items():
        # include time as the first dimension and make it go from 0 to 1
        residuals = [np.linspace(0, 1, len(eeg))]
        for channel1, channel2 in pair:
            residual = transform_residuals(eeg[channel1] - eeg[channel2], scaler_type)
            residuals.append(residual)
        brain_regions.append(np.stack(residuals).T)
    return np.stack(brain_regions)


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut=0.1, highcut=30, fs=200, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y

def preprocess_for_logsig(metadata, data_dir, scaler_type):
    """"Preprocess the eeg data to feed into the logsignature function.
        The output tensor is of the shape (paths_to_calculate x  path_length = 10000 x path_dimensions = 5).
        paths to calculate = number_of_eeg_recordings * 4 brain regions for each recording.
    """
    preprocessed = []
    for i, data in metadata.iterrows():
        eeg_id = data.eeg_id
        # eeg is sampled at 200 Hz
        offset = int(data.eeg_offset_seconds * 200)
        parquet_path = (f"{data_dir}{eeg_id}.parquet")
        # clip roughly the top and bottom 1% of the data
        eeg = pd.read_parquet(parquet_path).fillna(0).clip(-1300,2800)
        # bandpass filter
        eeg = pd.DataFrame(butter_bandpass_filter(eeg), columns=eeg.columns)
        eeg = eeg.iloc[offset:offset+10000]
        residuals = get_residuals(eeg, scaler_type)      
        preprocessed.append(residuals)
    preprocessed = np.concatenate(preprocessed, axis=0)
    
    return preprocessed

def calculate_logsignature(preprocessed, truncation_level):
    # assumes a 5 dimensional path
    logsignature = signatory.logsignature(preprocessed, truncation_level)
    return logsignature

def calculate_signature(preprocessed, truncation_level):
    # assumes a 5 dimensional path
    signature = signatory.signature(preprocessed, truncation_level)
    return signature

def calculate_logsignature_for_metadata(metadata, input_data_dir, output_data_dir, scaler_type, batch_size=100, device="cpu"):
    """Saves batches of tensors of the shape (batch_size x 4 (brain regions) x 829 (signature size))."""
    for i in range(0, len(metadata), batch_size):
        preprocessed = preprocess_for_logsig(metadata[i:i+batch_size], input_data_dir, scaler_type)
        preprocessed = torch.tensor(preprocessed, dtype=torch.float64).to(device)
        logsigs = calculate_logsignature(preprocessed, truncation_level=5).cpu()
        # 829 is the size of the logsignature for 5 dimensions and a truncation level 5
        logsigs = logsigs.reshape(-1,4,829)
        torch.save(logsigs, f"{output_data_dir}logsigs_lvl_5_scaler_{scaler_type}_{i}.pt")
        if i % (batch_size) == 0:
            print(f"Processed {i} records.")

def calculate_signature_for_metadata(metadata, input_data_dir, output_data_dir, scaler_type, batch_size=100, device="cpu"):
    """Saves batches of tensors of the shape (batch_size x 4 (brain regions) x 3905 (signature size))."""
    for i in range(0, len(metadata), batch_size):
        preprocessed = preprocess_for_logsig(metadata[i:i+batch_size], input_data_dir, scaler_type)
        preprocessed = torch.tensor(preprocessed, dtype=torch.float64).to(device)
        sigs = calculate_signature(preprocessed, truncation_level=5).cpu()
        # 3905 is the size of the signature for 5 dimensions
        sigs = sigs.reshape(-1,4,3905)
        torch.save(sigs, f"{output_data_dir}sigs_lvl_5_scaler_{scaler_type}_{i}.pt")
        if i % (batch_size) == 0:
            print(f"Processed {i} records.")