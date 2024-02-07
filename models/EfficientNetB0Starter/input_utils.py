import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from constants import TARGETS

# TODO: Why is this not in its own file? I guess it is computed very quickly.
def create_modified_eeg_metadata_df(eeg_metadata_df):
    """Create a new metadata file which integrates all sub_eeg_id."""

    # For each eeg_id (recall that this is the full recording)
    # - We take the associated spectrogram_id
    # - We find the min offset in the spectrogram
    # - Same with the max
    # - Get the patient_id
    # - Then below we compute the expert vote distribution
    train = eeg_metadata_df.groupby("eeg_id").agg(
        spec_id = pd.NamedAgg("spectrogram_id", "first"),
        min_offset_seconds = pd.NamedAgg("spectrogram_label_offset_seconds", "min"),
        max_offset_seconds = pd.NamedAgg("spectrogram_label_offset_seconds", "max"),
        patient_id = pd.NamedAgg("patient_id", "first"),
        target = pd.NamedAgg("expert_consensus", "first")
        )

    # TODO: Investigate further. Surely the vote distribution could change over time.
    # And if we're only doing the first eeg segment then this number is not representative
    # On the other hand this is only one recording and this can be seen as a
    # long term average and perhaps more representative of what the doctor will diagnose.
    # Even though the test only has one eeg_segment, perhaps doctor saw more

    # For each eeg_id, we have the total number of votes across all eeg segments
    # Why not in above groupby: Easier to do a for loop due to additional logic
    total_votes = eeg_metadata_df.groupby("eeg_id")[TARGETS].agg("sum")
    # Normalise so values represent a proportion (which sum to 1)
    total_votes = total_votes.div(total_votes.sum(axis=1), axis=0)
    for vote_label in TARGETS:
        train[vote_label] = total_votes[vote_label]

    # Have an index of 0, 1, ..., N as opposed to eeg_id
    train = train.reset_index()

    return train


class BaseDataset(Dataset):
    """Gives spectrograms."""
    def __init__(self, metadata, transform=None,
                 batch_size=32, shuffle=False, augment=False,
                 mode="train"):

        # Core data
        self.metadata = metadata

        self.batch_size = batch_size
        self.shuffle = shuffle
        # TODO: Not implemented yet, will need albumentations
        self.augment = augment
        # TODO: Do we need to worry about the mode? Doesn't PL handle that?
        self.mode = mode

        if transform is not None:
            self.transform = transform

        self.TARGETS = TARGETS
        self.num_targets = len(TARGETS)

        # TODO: Why 128 x 256?
        # My guess is that this is a standard size for an image
        # And we are using a standard Architecture which probably assumes 128x256

        # There are 100 frequencies, pad either side with 0s to get 128
        self.img_width  = 128
        # There are 300 timesteps (each 2 seconds) and crop by 22 symmetrically
        # to get 256
        self.img_height = 256

        # There are 4 types: Left and Right for Parasagittal and Lateral
        # Then 8 in total because using Kaggle Spectrograms and one inferred from EEG
        self.num_spectrograms = 8

    def __len__(self):
        return len(self.metadata)

    # TODO: Should this be done in the training loop
    # as opposed to the DataLoader?
    @staticmethod
    def transform(img):
        """Can be a single image (width, height)
        But can also be a (width, height, channels)
        """
        # LOG TRANSFORM SPECTROGRAM
        img = np.clip(img, np.exp(-4),np.exp(8))
        img = np.log(img)

        # For computing the mean and std
        num_channels = img.shape[-1] if len(img.shape) == 3 else 1
        reshaped_image = img.reshape(-1, num_channels)

        # STANDARDIZE PER IMAGE
        jitter = 1e-6
        mean_by_channel = np.nanmean(reshaped_image, axis=0)
        std_by_channel  = np.nanstd (reshaped_image, axis=0)
        img = (img - mean_by_channel) / (std_by_channel + jitter)
        img = np.nan_to_num(img, nan=0.0)

        return img

    # TODO: Need to test this.
    def __getitem__(self, idx):
        "Generates data containing batch_size samples"
        # idx will contain the indices for our batch
        # TODO: Check if we will ever use integer index as opposed to a list
        inferred_batch_size = len(idx)

        # Shape is (batch_size, width, height, num_channels)
        # TODO: PyTorch uses channels first.
        # Keep like this for downstream compatibility. But change to faster one.
        # (Between channel first and last)

        X = np.zeros((inferred_batch_size, self.img_width, self.img_height, self.num_spectrograms), dtype="float32")
        y = np.zeros((inferred_batch_size, self.num_targets), dtype="float32")

        rows = self.metadata.iloc[idx]

        # NOTE: Have to do a for loop because the full spectrograms
        # may be different length

        for i in range(inferred_batch_size):
            row = rows.iloc[i]
            # TODO: Why are they looking at 1/4 of the offset range?
            # To bias more towards the start?
            # Surely you should just use min_offset_seconds.
            # TODO: Maybe divided by 2 to get in terms of seconds? That would be wrong
            offset = int((row["min_offset_seconds"] + row["max_offset_seconds"]) // 4)

            spectrogram_id = row.spec_id
            eeg_spectrogram_id = row.eeg_id

            # EXTRACT 300 ROWS OF SPECTROGRAM
            # Spectrogram sampled every 2 seconds soe 300 rows is 600 seconds
            # I.e. 10 minutes
            # NOTE: In actual parquet file, the first column is time.
            # But precomputed specs does not have a time column.
            # Shape progression
            # (300, 100 * 4)
            # (300, 4, 100)
            # (300, 100, 4)
            # (100, 300, 4)
            kaggle_spectrograms = \
                self.get_spectrogram(spectrogram_id)[offset: offset+300] \
                    .reshape(300, 4, 100) \
                    .swapaxes(1, 2) \
                    .swapaxes(0, 1)
            # Shape: (num_timesteps, num_frequencies, num_spectrograms)
            kaggle_spectrograms = self.transform(kaggle_spectrograms)

            # CROP TO 256 TIME STEPS with 22:-22
            # Pad to 128 frequencies with 14:-14
            # TODO: Why divide value by 2?
            X[i, 14:-14, :, :4] = kaggle_spectrograms[..., 22:-22, :] / 2.0

            # EEG SPECTROGRAMS, already processed?
            eeg_spectrograms = self.get_eeg_spectrogram(eeg_spectrogram_id)
            X[i, :, :, 4:] = eeg_spectrograms

            y[i] = row[TARGETS]

        return X, y

    def get_spectrogram(self, spectrogram_id):
        pass

    def get_eeg_spectrogram(self, eeg_spectrogram_id):
        pass


# TODO: Maybe consider doing an LRU sort of thing?
# But actually most likely just going to have enough resources to
# hold everything in RAM and so we use PreloadedDataset for speed
class LazyDataset(BaseDataset):
    """Reads te spectrograms from path"""
    def __init__(self, metadata, spectrogram_dir, eeg_spectrogram_dir,
                 transform=None, batch_size=32, shuffle=False, augment=False,
                 mode="train"):

        self.spectrogram_dir = spectrogram_dir
        self.eeg_spectrogram_dir = eeg_spectrogram_dir

        super().__init__(metadata, transform,
                         batch_size, shuffle, augment, mode)

    def get_spectrogram(self, spectrogram_id):
        spectrogram = pd.read_parquet(
            f"{self.spectrogram_dir}/{spectrogram_id}.parquet")
        # Get rid of the time column (first column)
        # and convert to numpy
        spectrogram = spectrogram.iloc[:, 1:].to_numpy()
        return spectrogram

    def get_eeg_spectrogram(self, eeg_spectrogram_id):
        eeg_spectrogram = np.load(
            f"{self.eeg_spectrogram_dir}/{eeg_spectrogram_id}.npy")
        return eeg_spectrogram


class PreloadedDataset(BaseDataset):
    """Has spectrogram dictionaries which have the data preloaded"""
    def __init__(self, metadata, spectrograms, eeg_spectrograms,
                 transform=None, batch_size=32, shuffle=False, augment=False,
                 mode="train"):

        self.spectrograms = spectrograms
        self.eeg_spectrograms = eeg_spectrograms

        super().__init__(metadata, transform,
                         batch_size, shuffle, augment, mode)

    def get_spectrogram(self, spectrogram_id):
        return self.spectrograms[spectrogram_id]

    def get_eeg_spectrogram(self, eeg_spectrogram_id):
        return self.eeg_spectrograms[eeg_spectrogram_id]
