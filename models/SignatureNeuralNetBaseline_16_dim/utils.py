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
    num_votes = metadata.iloc[:, -6:].sum(axis=1)
    metadata = metadata[(num_votes >= 10)]
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
    elif scaler_type.startswith("meanvarPerChannel"):
        scaler_std = float(scaler_type.split("_")[1])
        scaler = TimeSeriesScalerMeanVariance(std=scaler_std)
        ts = scaler.fit_transform(ts, std=scaler_std)
    elif scaler_type.startswith("constant"):
        scaler_constant = float(scaler_type.split("_")[1])
        ts = ts / scaler_constant
    elif scaler_type.startswith("meanvar"):
        # this is done later since we atke the variance across all channels
        pass
    else:
        raise ValueError(f"Unknown scaler type {scaler_type}")
    return ts

def transform_residuals(residuals, scaler_type):
    residuals = rescale(residuals.values.reshape(1,-1,1), scaler_type).reshape(-1)
    return residuals

def get_residuals(eeg, scaler_type):
    """Doctors look at the difference between two neighboring channels.
       Calculate the residuals for each channel pair.
       Group by brain region."""
    brain_regions = []
    for region, pair in RESIDUAL_PAIRS.items():
        # include time as the first dimension and make it go from 0 to 1
        residuals = []
        for channel1, channel2 in pair:
            residual = transform_residuals(eeg[channel1] - eeg[channel2], scaler_type)
            residuals.append(residual)
        brain_regions.append(np.stack(residuals).T)
    brain_regions = np.stack(brain_regions)

    if scaler_type.startswith("meanvar"):
        brain_regions = brain_regions - brain_regions.mean(axis=1, keepdims=True)
        brain_regions = brain_regions / (brain_regions.std()+1e-6)
    return brain_regions.clip(-4, 4)

def augment_with_time(residuals):
    """ take residuals of the shape (4, 10000, 4) and augment with time to obtain (4, 10000, 5)"""
    augmented_regions = []
    for region_index in range(4):
        augmented_regions.append(np.concatenate([residuals[region_index], np.linspace(0,1,10000).reshape(-1,1)], axis=1))
    return np.stack(augmented_regions)

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

def preprocess_for_sig(metadata, data_dir, scaler_type):
    """"Preprocess the eeg data to feed into the logsignature function.
        The output tensor is of the shape (paths_to_calculate x  path_length = 10000 x path_dimensions = 5).
        paths to calculate = number_of_eeg_recordings * 4 brain regions for each recording.
    """
    preprocessed = []
    for i, data in metadata.iterrows():
        eeg_id = data.eeg_id
        # eeg is sampled at 200 Hz
        offset = int(data.eeg_offset_seconds * 200 )
        parquet_path = (f"{data_dir}{eeg_id}.parquet")
        eeg = pd.read_parquet(parquet_path)
        # replace 9999 with 0
        eeg = eeg.replace(9999, 0)
        eeg = eeg.fillna(0).clip(-1000,1000)
        eeg = eeg.iloc[offset:offset+10000]
        # bandpass filter
        eeg = pd.DataFrame(butter_bandpass_filter(eeg), columns=eeg.columns)
        residuals = get_residuals(eeg, scaler_type)
        residuals = augment_with_time(residuals)      
        preprocessed.append(residuals)
    preprocessed = np.concatenate(preprocessed, axis=0)
    
    return preprocessed

def preprocess_for_sig_test(metadata, data_dir, scaler_type):
    """Preprocessing needs to be different for the kaggle test set since we only have 50 second eeg recordings."""
    preprocessed = []
    for i, data in metadata.iterrows():
        eeg_id = data.eeg_id
        parquet_path = (f"{data_dir}{eeg_id}.parquet")
        eeg = pd.read_parquet(parquet_path)
        # replace 9999 with 0
        eeg = eeg.replace(9999, 0)
        eeg = eeg.fillna(0).clip(-1000,1000)
        eeg = pd.DataFrame(butter_bandpass_filter(eeg), columns=eeg.columns)
        residuals = get_residuals(eeg, scaler_type)
        residuals = augment_with_time(residuals)      
        preprocessed.append(residuals)
    preprocessed = np.concatenate(preprocessed, axis=0)

    return preprocessed

def calculate_logsignature(preprocessed, truncation_level):
    logsignature = signatory.logsignature(preprocessed, truncation_level)
    return logsignature

def calculate_signature(preprocessed, truncation_level):
    signature = signatory.signature(preprocessed, truncation_level)
    return signature

def calculate_logsignature_for_metadata_test(metadata, input_data_dir, scaler_type, device="cpu", level=5):
    """Return the tensor of calculated signtures.
       Use this function to calculate the logsignatures for the kaggle test set.
    """
    preprocessed = preprocess_for_sig_test(metadata, input_data_dir, scaler_type)
    preprocessed = torch.tensor(preprocessed, dtype=torch.float64).to(device)
    logsigs = calculate_logsignature(preprocessed, truncation_level=level).cpu()
    size = logsigs.shape[1]
    logsigs = logsigs.reshape(-1,4,size)
    return logsigs


def calculate_signature_for_metadata_test(metadata, input_data_dir, scaler_type, device="cpu", level=5):
    """Return the tensor of calculated signtures.
       Use this function to calculate the signatures for the kaggle test set.
    """
    preprocessed = preprocess_for_sig_test(metadata, input_data_dir, scaler_type)
    preprocessed = torch.tensor(preprocessed, dtype=torch.float64).to(device)
    sigs = calculate_signature(preprocessed, truncation_level=level).cpu()
    size = sigs.shape[1]
    sigs = sigs.reshape(-1,4,size)
    return sigs

