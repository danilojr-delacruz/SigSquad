import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import eeg_preprocessing_transform
from constants import TARGETS, CHANNELS, NUM_CHANNELS, EEG_SNAPSHOT_DURATION


def get_eeg_window(file):
    """Return cropped EEG window.

    Default setting is to return the middle 50-sec window.
    over the entire recording

    Args:
        file: EEG file path
        test: if True, there's no need to truncate EEGs

    Returns:
        eeg_win: cropped EEG window
    """
    eeg = pd.read_parquet(file, columns=CHANNELS)
    n_pts = len(eeg)
    offset = (n_pts - EEG_SNAPSHOT_DURATION) // 2
    eeg = eeg.iloc[offset:offset + EEG_SNAPSHOT_DURATION]

    eeg_win = np.zeros((EEG_SNAPSHOT_DURATION, NUM_CHANNELS))
    for j, col in enumerate(CHANNELS):
        eeg_raw = eeg[col].values.astype("float32")
        eeg_win[:, j] = eeg_raw

    return eeg_win


# For training good if we can preload
# But for inference it is not necessary as we are only making one pass
def preload_precomputed_eeg(eeg_dir):
    eegs = np.load(eeg_dir, allow_pickle=True).item()
    return eegs


# Don't need to precompute as cheap to compute
def modify_train_metadata(train_metadata):
    """Create a new metadata file which integrates all sub_eeg_id."""

    # For each eeg_id (recall that this is the full recording)
    # - We take the associated spectrogram_id
    # - We find the min offset in the spectrogram
    # - Same with the max
    # - Get the patient_id
    # - Then below we compute the expert vote distribution
    train = train_metadata.copy()
    num_votes = train.iloc[:, -6:].sum(axis=1)
    train = train[num_votes >= 10]

    train = train.groupby("eeg_id").agg(
        spectrogram_id     = pd.NamedAgg("spectrogram_id", "first"),
        min_offset_seconds = pd.NamedAgg("eeg_label_offset_seconds", "min"),
        max_offset_seconds = pd.NamedAgg("eeg_label_offset_seconds", "max"),
        patient_id         = pd.NamedAgg("patient_id", "first"),
        target             = pd.NamedAgg("expert_consensus", "first")
        )

    # Use snapshot in middle of recording
    train["offset_seconds"] = (train["min_offset_seconds"] + train["max_offset_seconds"]) // 2
    train = train.drop(["min_offset_seconds", "max_offset_seconds"], axis=1)

    # TODO: Investigate further. Surely the vote distribution could change over time.
    # And if we're only doing the first eeg segment then this number is not representative
    # On the other hand this is only one recording and this can be seen as a
    # long term average and perhaps more representative of what the doctor will diagnose.
    # Even though the test only has one eeg_segment, perhaps doctor saw more

    # For each eeg_id, we have the total number of votes across all eeg segments
    # Why not in above groupby: Easier to do a for loop due to additional logic
    total_votes = train_metadata.groupby("eeg_id")[TARGETS].agg("sum")
    # Normalise so values represent a proportion (which sum to 1)
    total_votes = total_votes.div(total_votes.sum(axis=1), axis=0)
    for vote_label in TARGETS:
        train[vote_label] = total_votes[vote_label]

    # Have an index of 0, 1, ..., N as opposed to eeg_id
    train = train.reset_index()

    return train


class TrainDataset(Dataset):
    """Gives spectrograms."""
    def __init__(self, metadata, fetcher, shuffle=False):

        # Core data
        self.metadata = metadata
        self.fetcher  = fetcher

        self.shuffle = shuffle

        self.TARGETS = TARGETS
        self.num_targets = len(TARGETS)

    def __len__(self):
        return len(self.metadata)


    @staticmethod
    def transform(x):
        return eeg_preprocessing_transform(x)

    def __getitem__(self, idx):
        """Assuming idx is an int."""

        row = self.metadata.iloc[idx]
        eeg_id = row.eeg_id

        # TODO: Assuming for now that the eeg is uniquely specified by the id
        # and is exactly 50 seconds long.
        # True for the test case, but limiting for training.
        X = self.transform(self.fetcher.get_eeg(eeg_id))
        y = row[TARGETS].values.astype(float)

        # Convert into tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y


class ValidationDataset(TrainDataset):
    # Needs to read an offset as still working with
    # train metadata
    # Whereas test does not need this.
    # However only want it to return X
    def __getitem__(self, idx):
        X, y = super().__getitem__(idx)
        return X


# NOTE: Train and Test set are sufficiently different to warrant different loaders
# TODO: Shares a lot of code with TrainDataset, maybe inherit from base class
class TestDataset(Dataset):
    """Gives spectrograms."""
    def __init__(self, metadata, fetcher, shuffle=False):

        # Core data
        self.metadata = metadata
        self.fetcher  = fetcher

        self.shuffle = shuffle

        self.TARGETS = TARGETS
        self.num_targets = len(TARGETS)

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def transform(x):
        return eeg_preprocessing_transform(x)

    def __getitem__(self, idx):
        """Assuming idx is an int."""

        row = self.metadata.iloc[idx]
        eeg_id = row.eeg_id

        # TODO: Assuming for now that the eeg is uniquely specified by the id
        # and is exactly 50 seconds long.
        # True for the test case, but limiting for training.
        X = self.transform(self.fetcher.get_eeg(eeg_id))

        # Convert into tensors
        X = torch.tensor(X, dtype=torch.float32)

        return X


# Have in name only for now
class Fetcher:
    """Fetches data"""
    pass


class LazyEegFetcher(Fetcher):
    def __init__(self, eeg_dir):
        self.eeg_dir = eeg_dir

    def get_eeg(self, eeg_id):
        eeg = get_eeg_window(f"{self.eeg_dir}/{eeg_id}.parquet")
        return eeg


class PreloadedEegFetcher(Fetcher):
    """Has spectrogram dictionaries which have the data preloaded"""
    def __init__(self, eeg_dir):
        self.eegs = preload_precomputed_eeg(eeg_dir)

    def get_eeg(self, eeg_id):
        return self.eegs[eeg_id]
