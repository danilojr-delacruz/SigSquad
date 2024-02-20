import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from constants import TARGETS

# Don't need to precompute as cheap to compute
def modify_train_metadata(train_metadata):
    """Create a new metadata file which integrates all sub_eeg_id."""

    # For each eeg_id (recall that this is the full recording)
    # - We take the associated spectrogram_id
    # - We find the min offset in the spectrogram
    # - Same with the max
    # - Get the patient_id
    # - Then below we compute the expert vote distribution
    train = train_metadata.groupby("eeg_id").agg(
        spec_id            = pd.NamedAgg("spectrogram_id", "first"),
        min_offset_seconds = pd.NamedAgg("spectrogram_label_offset_seconds", "min"),
        max_offset_seconds = pd.NamedAgg("spectrogram_label_offset_seconds", "max"),
        patient_id         = pd.NamedAgg("patient_id", "first"),
        target             = pd.NamedAgg("expert_consensus", "first")
        )

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


# TODO: Add some tests to check these work.
def create_spectrogram_image_tile(spectrograms):
    """Assuming shape is tensor of shape
    (num_spectrograms, height, width)
    (8               , 128   , 256  )
    Where the first four is the kaggle spectrograms, followed by EEG

    # Output is
    # ---> Axis 2
    # K1 E1
    # K2 E2
    # K3 E3
    # K4 E4
    # Vertical is axis 1, horizontal is axis 2.
    # Axis 3 is colour, which is gray-scale as values repeated.
    # Summary: Input is just all the spectrograms tiled together into one
    # (3, 512, 512) image.
    """

    num_spectrograms, height, width = spectrograms.shape
    # Assumption, manual override
    if num_spectrograms != 8:
        raise Exception(f"Num spectrograms = {num_spectrograms}, expected 8")

    num_spectrograms = 8

    # Initial Shape: (8, 128, 256)
    # (2, 512, 256)
    X = spectrograms.reshape(2, 4*height, width)
    # Horizontally stack the two (long) columns
    # (512, 512)
    X = torch.concatenate([X[0, ...], X[1, ...]], axis=-1)
    # Repeat along the third dimension
    X = X.unsqueeze(0).repeat(3, 1, 1)

    return X


class TrainDataset(Dataset):
    """Gives spectrograms."""
    def __init__(self, metadata, fetcher, transform=None,
                 shuffle=False, augment=False):

        # Core data
        self.metadata = metadata
        self.fetcher  = fetcher

        self.shuffle = shuffle
        # TODO: Not implemented yet, will need albumentations
        self.augment = augment

        if transform is not None:
            self.transform = transform

        self.TARGETS = TARGETS
        self.num_targets = len(TARGETS)

        # 128 x 256 to get into standard shape for EfficientNet

        # There are 100 frequencies, pad either side with 0s to get 128
        self.img_height  = 128
        # There are 300 timesteps (each 2 seconds)
        # crop by 22 symmetrically to get 256
        self.img_width   = 256

        # There are 4 types: Left and Right for Parasagittal and Lateral
        # Then 8 in total because using Kaggle Spectrograms and one inferred from EEG
        self.num_spectrograms = 8

    def __len__(self):
        return len(self.metadata)

    # TODO: Should this be done in the training loop
    # as opposed to the DataLoader?
    @staticmethod
    def transform(img):
        """Assumed to be (channels, height, width)"""
        # LOG TRANSFORM SPECTROGRAM
        img = np.clip(img, np.exp(-4),np.exp(8))
        img = np.log(img)

        # For computing the mean and std
        num_channels = img.shape[0]
        reshaped_image = img.reshape(num_channels, -1)

        # STANDARDIZE PER IMAGE
        jitter = 1e-6
        mean_by_channel = np.nanmean(reshaped_image, axis=1) \
                            .reshape(num_channels, 1, 1)
        std_by_channel  = np.nanstd (reshaped_image, axis=1) \
                            .reshape(num_channels, 1, 1)

        img = (img - mean_by_channel) / (std_by_channel + jitter)
        img = np.nan_to_num(img, nan=0.0)

        return img

    # TODO: Need to test this.
    def __getitem__(self, idx):
        """Assuming idx is an int."""

        # NOTE: PyTorch Layout of (Channel, Height, Width)
        X = np.zeros((self.num_spectrograms, self.img_height, self.img_width), dtype="float32")
        y = np.zeros((self.num_targets,), dtype="float32")

        row = self.metadata.iloc[idx]

        # TODO: Why are they looking at 1/4 of the offset range?
        # To bias more towards the start?
        # Surely you should just use min_offset_seconds.
        # TODO: Maybe divided by 2 to get in terms of seconds? That would be wrong
        offset = int((row["min_offset_seconds"] + row["max_offset_seconds"]) // 4)

        spectrogram_id = row.spec_id
        eeg_spectrogram_id = row.eeg_id

        # EXTRACT 300 ROWS OF SPECTROGRAM
        # Spectrogram sampled every 2 seconds so 300 rows is 600 seconds
        # I.e. 10 minutes
        # NOTE: In actual parquet file, the first column is time.
        # But precomputed specs does not have a time column.
        # Shape progression
        # (300, 100 * 4)
        # (300, 4, 100)
        # (4, 300, 100)
        # (4, 100, 300)
        kaggle_spectrograms = \
            self.fetcher.get_spectrogram(spectrogram_id)[offset: offset+300] \
                .reshape(300, 4, 100) \
                .swapaxes(0, 1) \
                .swapaxes(1, 2)
        # Shape: (num_spectrograms, num_timesteps, num_frequencies)
        kaggle_spectrograms = self.transform(kaggle_spectrograms)

        # CROP TO 256 TIME STEPS with 22:-22
        # Pad to 128 frequencies with 14:-14
        # TODO: Why divide value by 2?
        X[:4, 14:-14, :] = kaggle_spectrograms[:, :, 22:-22] / 2.0

        # EEG SPECTROGRAMS, already processed?
        # Initial Shape: (128, 256, 4)
        # Move into    : (4, 128, 256)
        # TODO: When you have control over this, have the shape as (300, 400)
        # for consistency
        eeg_spectrograms = self.fetcher.get_eeg_spectrogram(eeg_spectrogram_id) \
                               .swapaxes(2, 1) \
                               .swapaxes(0, 1)
        X[4:, ...] = eeg_spectrograms

        y = row[TARGETS].values.astype(float)

        # Convert into tensors
        X = create_spectrogram_image_tile(torch.tensor(X))
        y = torch.tensor(y, dtype=torch.float32)

        return X, y


# NOTE: Train and Test set are sufficiently different to warrant different loaders
class TestDataset(Dataset):
    """Gives spectrograms."""
    def __init__(self, metadata, fetcher, transform=None,
                 shuffle=False, augment=False):

        # Core data
        self.metadata = metadata
        self.fetcher  = fetcher

        self.shuffle = shuffle
        # TODO: Not implemented yet, will need albumentations
        self.augment = augment

        if transform is not None:
            self.transform = transform

        self.TARGETS = TARGETS
        self.num_targets = len(TARGETS)

        # 128 x 256 to get into standard shape for EfficientNet

        # There are 100 frequencies, pad either side with 0s to get 128
        self.img_height  = 128
        # There are 300 timesteps (each 2 seconds)
        # crop by 22 symmetrically to get 256
        self.img_width   = 256

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
        """Assuming idx is an int."""

        # NOTE: PyTorch Layout of (Channel, Height, Width)
        X = np.zeros((self.num_spectrograms, self.img_height, self.img_width), dtype="float32")

        row = self.metadata.iloc[idx]

        spectrogram_id = row.spec_id
        eeg_spectrogram_id = row.eeg_id

        # EXTRACT 300 ROWS OF SPECTROGRAM
        # Spectrogram sampled every 2 seconds so 300 rows is 600 seconds
        # I.e. 10 minutes
        # NOTE: In actual parquet file, the first column is time.
        # But precomputed specs does not have a time column.
        # Shape progression
        # (300, 100 * 4)
        # (300, 4, 100)
        # (4, 300, 100)
        # (4, 100, 300)
        kaggle_spectrograms = \
            self.fetcher.get_spectrogram(spectrogram_id) \
                .reshape(300, 4, 100) \
                .swapaxes(0, 1) \
                .swapaxes(1, 2)
        # Shape: (num_spectrograms, num_timesteps, num_frequencies)
        kaggle_spectrograms = self.transform(kaggle_spectrograms)

        # CROP TO 256 TIME STEPS with 22:-22
        # Pad to 128 frequencies with 14:-14
        # TODO: Why divide value by 2?
        X[:4, 14:-14, :] = kaggle_spectrograms[:, :, 22:-22] / 2.0

        # EEG SPECTROGRAMS, already processed?
        eeg_spectrograms = self.fetcher.get_eeg_spectrogram(eeg_spectrogram_id)
        X[4:, ...] = eeg_spectrograms

        # Convert into tensors
        X = create_spectrogram_image_tile(torch.tensor(X))

        return X


# Have in name only for now
class Fetcher:
    """Fetches data"""
    pass


# TODO: Maybe consider doing an LRU sort of thing?
# But actually most likely just going to have enough resources to
# hold everything in RAM and so we use PreloadedDataset for speed
class LazySpectrogramFetcher(Fetcher):
    """Reads the spectrograms from path.
    Assumes that eeg_spectrograms are precomputed and stored.
    """
    def __init__(self, spectrogram_dir, eeg_spectrogram_dir):

        self.spectrogram_dir = spectrogram_dir
        self.eeg_spectrogram_dir = eeg_spectrogram_dir

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


class PreloadedSpectrogramFetcher(Fetcher):
    """Has spectrogram dictionaries which have the data preloaded"""
    def __init__(self, spectrograms, eeg_spectrograms):

        self.spectrograms = spectrograms
        self.eeg_spectrograms = eeg_spectrograms

    def get_spectrogram(self, spectrogram_id):
        return self.spectrograms[spectrogram_id]

    def get_eeg_spectrogram(self, eeg_spectrogram_id):
        return self.eeg_spectrograms[eeg_spectrogram_id]
