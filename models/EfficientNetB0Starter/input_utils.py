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
        min = pd.NamedAgg("spectrogram_label_offset_seconds", "min"),
        max = pd.NamedAgg("spectrogram_label_offset_seconds", "max"),
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


# Refer to this to understand Dataset
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class DataGenerator(Dataset):
    "Generates data for Keras"
    def __init__(self, data, specs, eeg_specs, transform=None,
                 batch_size=32, shuffle=False, augment=False, mode="train"):

        # Core data
        self.data = data
        self.specs = specs
        self.eeg_specs = eeg_specs

        self.batch_size = batch_size
        self.shuffle = shuffle
        # TODO: Not implemented yet, will need albumentations
        self.augment = augment
        # TODO: Do we need to worry about the mode? Doesn't PL handle that?
        self.mode = mode

        self.transform = transform

        self.TARGETS = TARGETS
        self.num_targets = len(TARGETS)

        self.img_width  = 128
        self.img_height = 256

        # There are 4 types: Left and Right for Parasagittal and Lateral
        # Then 8 in total because using Kaggle Spectrograms and one inferred from EEG
        self.num_spectrograms = 8

    def __len__(self):
        return len(self.data)

    # TODO: Should this be done in the training loop
    # as opposed to the DataLoader?
    def transform(self, img):
        # LOG TRANSFORM SPECTROGRAM
        img = np.clip(img,np.exp(-4),np.exp(8))
        img = np.log(img)

        # STANDARDIZE PER IMAGE
        ep = 1e-6
        m = np.nanmean(img.flatten())
        s = np.nanstd(img.flatten())
        img = (img-m)/(s+ep)
        img = np.nan_to_num(img, nan=0.0)

        return img

    def __getitem__(self, idx):
        "Generates data containing batch_size samples"

        # idx will contain the indices for our batch
        # Shape is (batch_size, )
        X = np.zeros((self.batch_size, self.img_width, self.img_height, self.num_spectrograms), dtype="float32")
        y = np.zeros((self.batch_size, self.num_targets), dtype="float32")
        img = np.ones((self.img_width, self.img_height),dtype="float32")

        for j,i in enumerate(idx):
            row = self.data.iloc[i]
            # if self.mode=="test":
            #     r = 0
            r = int( (row["min"] + row["max"])//4 )

            for k in range(4):
                # EXTRACT 300 ROWS OF SPECTROGRAM
                img = self.transform(self.specs[row.spec_id][r:r+300,k*100:(k+1)*100].T)

                # CROP TO 256 TIME STEPS
                X[j,14:-14,:,k] = img[:,22:-22] / 2.0

            # EEG SPECTROGRAMS
            img = self.eeg_specs[row.eeg_id]
            X[j,:,:,4:] = img

            if self.mode!="test":
                y[j,] = row[self.TARGETS]

        return X,y