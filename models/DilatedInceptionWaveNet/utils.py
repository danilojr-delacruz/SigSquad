import pandas as pd
import numpy as np

from scipy.signal import butter, lfilter
from constants import EEG_SNAPSHOT_DURATION, NUM_CHANNELS, FEAT2CODE


def montage_difference(x):
    """Generate features based on Chris' magic formula."""
    x_tmp = np.zeros((EEG_SNAPSHOT_DURATION, NUM_CHANNELS), dtype="float32")

    # Generate features
    x_tmp[:, 0] = x[:, FEAT2CODE["Fp1"]] - x[:, FEAT2CODE["T3"]]
    x_tmp[:, 1] = x[:, FEAT2CODE["T3"]] - x[:, FEAT2CODE["O1"]]

    x_tmp[:, 2] = x[:, FEAT2CODE["Fp1"]] - x[:, FEAT2CODE["C3"]]
    x_tmp[:, 3] = x[:, FEAT2CODE["C3"]] - x[:, FEAT2CODE["O1"]]

    x_tmp[:, 4] = x[:, FEAT2CODE["Fp2"]] - x[:, FEAT2CODE["C4"]]
    x_tmp[:, 5] = x[:, FEAT2CODE["C4"]] - x[:, FEAT2CODE["O2"]]

    x_tmp[:, 6] = x[:, FEAT2CODE["Fp2"]] - x[:, FEAT2CODE["T4"]]
    x_tmp[:, 7] = x[:, FEAT2CODE["T4"]] - x[:, FEAT2CODE["O2"]]

    return x_tmp


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)

    return filtered_data


# TODO: What is the mu law encoding for?
# Original author chose to set it as false, so ignore for now
def quantize_data(self, data, classes):
    mu_x = self._mu_law_encoding(data, classes)

    return mu_x


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)

    return mu_x


def eeg_preprocessing_transform(x):
    """Assumed the shape is (L, C)"""
    # Replace any 9999 filler value into nan
    x = x.copy()
    x[x == 9999] = np.nan

    # Change into Double Banana Montage
    x = montage_difference(x)

    # Centre at 0
    # Shape is (1, C')
    median_x = np.median(x, axis=0)[:, None]
    x = x - median_x

    # Standard Preprocessing
    x = np.clip(x, -1024, 1024)
    x = np.nan_to_num(x, nan=0)

    # TODO: Don't want to perform instance normalisation
    # Want to be able to compare between EEGs
    x = x / 32.0

    # Filter frequencies above 20 Hz
    x = butter_lowpass_filter(x)

    # TODO: Do not perform mu law encoding yet
    # x = mu_law_encoding(x)

    # Down sample
    downsampling_rate = 5
    x = x[::downsampling_rate, :]

    return x