import iisignature
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

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

def transform_residuals(residuals):
    """Details taken from the preprocessing in https://www.kaggle.com/code/cdeotte/wavenet-starter-lb-0-52?scriptVersionId=160158478."""
    # standardize
    residuals = np.clip(residuals, -1024, 1024)
    residuals = np.nan_to_num(residuals, nan=0)/32
    # low pass filter
    residuals = butter_lowpass_filter(residuals)
    return residuals


def get_residuals(eeg):
    """Doctors look at the difference between two neighboring channels.
       Calculate the residuals for each channel pair.
       Group by brain region."""
    brain_regions = []
    for region, pair in RESIDUAL_PAIRS.items():
        # include time as the first dimension and make the range approximately match the range of the residuals
        residuals = [np.array(range(10000))/1000]
        for channel1, channel2 in pair:
            residual = transform_residuals(eeg[channel1] - eeg[channel2])
            residuals.append(residual)
        brain_regions.append(np.stack(residuals).T)
    return np.stack(brain_regions)

def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    """Filter out the frequencies above 20Hz.
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

def preprocess_for_logsig(metadata, data_dir):
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
        eeg = pd.read_parquet(parquet_path)
        # get 50 seconds of eeg
        eeg = eeg.iloc[offset:offset+10000]
        residuals = get_residuals(eeg)      
        preprocessed.append(residuals)
    preprocessed = np.concatenate(preprocessed, axis=0)
    
    return preprocessed

def calculate_logsignature(preprocessed, truncation_level=6):
    # assumes a 5 dimensional path
    s = iisignature.prepare(5, truncation_level)
    logsignature = iisignature.logsig(preprocessed, s)
    return logsignature

def calculate_logsignature_for_metadata(metadata, input_data_dir, output_data_dir, batch_size=100):
    """Saves batches of tensors of the shape (batch_size x 4 (brain regions) x 829 (signature size))."""
    for i in range(0, len(metadata), batch_size):
        preprocessed = preprocess_for_logsig(metadata[i:i+batch_size], input_data_dir)
        logsigs = calculate_logsignature(preprocessed, truncation_level=5)
        # 829 is the size of the logsignature for 5 dimensions and a truncation level 5
        logsigs = logsigs.reshape(-1,4,829)
        np.save(f"{output_data_dir}logsigs_lvl_5_{i}.npy", logsigs)