import pandas as pd
import torch

from torch.utils.data import Dataset


TRAIN_METADATA_DIR = "../../train.csv"

EEG_DIR             = "../../train_eegs"
SPECTROGRAM_DIR     = "../../train_spectrograms"

EEG_SAMPLING_RATE_HZ = 200
EEG_SNAPSHOT_DURATION_SECONDS = 50
EEG_SNAPSHOT_DURATION = EEG_SNAPSHOT_DURATION_SECONDS * EEG_SAMPLING_RATE_HZ

SPECTROGRAM_SAMPLING_RATE_HZ = 0.5
SPECTROGRAM_SNAPSHOT_DURATION_SECONDS = 600
SPECTROGRAM_SNAPSHOT_DURATION = int(SPECTROGRAM_SNAPSHOT_DURATION_SECONDS * SPECTROGRAM_SAMPLING_RATE_HZ)

NUM_BRAIN_REGIONS = 4
NUM_SPECTROGRAM_FREQUENCIES = 100

TARGETS = [
    "seizure_vote",	"lpd_vote", "gpd_vote",
    "lrda_vote", "grda_vote", "other_vote"
]


METADATA_DF = pd.read_csv(TRAIN_METADATA_DIR)


class RichDataset(Dataset):
    """Returns eeg, spectrograms as DataFrames
    Returns votes as Series

    Can only load one point at a time.
    Hence index with ints only.
    Index corresponds to those in METADATA_DF

    Dataloader will handle batching for us
    and it will be parallelised when we
    specify num_workers > 1
    """
    def __init__(self, transform=None):

        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(METADATA_DF)

    def get_eeg(self, eeg_id):
        filename = f"{EEG_DIR}/{eeg_id}.parquet"
        return pd.read_parquet(filename)

    def get_spectrogram(self, spectrogram_id):
        filename = f"{SPECTROGRAM_DIR}/{spectrogram_id}.parquet"
        return pd.read_parquet(filename)

    def __getitem__(self, idx):
        """Can only index with an integer"""
        assert type(idx) is int, "Can only index with int"

        row = METADATA_DF.iloc[idx]

        eeg_id = row.eeg_id
        spectrogram_id = row.spectrogram_id

        eeg_offset         = int(row.eeg_label_offset_seconds * EEG_SAMPLING_RATE_HZ)
        spectrogram_offset = int(row.spectrogram_label_offset_seconds * SPECTROGRAM_SAMPLING_RATE_HZ)

        eeg_full         = self.get_eeg(eeg_id)
        spectrogram_full = self.get_spectrogram(spectrogram_id)

        eeg_snapshot         = eeg_full.iloc[
            eeg_offset: eeg_offset + EEG_SNAPSHOT_DURATION, :]
        # Ignore the first column which is time
        spectrogram_snapshot = spectrogram_full.iloc[
            spectrogram_offset: spectrogram_offset + SPECTROGRAM_SNAPSHOT_DURATION, 1:]

        votes = row[TARGETS]

        # eeg_snapshot is (10_000, 20)
        # spectrogram_snapshot is (300, 400)
        # votes is (6, )
        return eeg_snapshot, spectrogram_snapshot, votes