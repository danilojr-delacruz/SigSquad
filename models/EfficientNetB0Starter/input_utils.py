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


# TODO: Migrate onto PyTorch Lightning
# Train Dataloader
class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"
    def __init__(self, data, specs, eeg_specs, TARGETS,
                 batch_size=32, shuffle=False, augment=False, mode="train"):

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        # TODO: Not implemented yet, will need albumentations
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.on_epoch_end()

        self.TARGETS = TARGETS

    def __len__(self):
        "Denotes the number of batches per epoch"
        ct = int( np.ceil( len(self.data) / self.batch_size ) )
        return ct

    def __getitem__(self, index):
        "Generate one batch of data"
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange( len(self.data) )
        if self.shuffle: np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        "Generates data containing batch_size samples"

        X = np.zeros((len(indexes),128,256,8),dtype="float32")
        y = np.zeros((len(indexes),6),dtype="float32")
        img = np.ones((128,256),dtype="float32")

        for j,i in enumerate(indexes):
            row = self.data.iloc[i]
            if self.mode=="test":
                r = 0
            else:
                r = int( (row["min"] + row["max"])//4 )

            for k in range(4):
                # EXTRACT 300 ROWS OF SPECTROGRAM
                img = self.specs[row.spec_id][r:r+300,k*100:(k+1)*100].T

                # LOG TRANSFORM SPECTROGRAM
                img = np.clip(img,np.exp(-4),np.exp(8))
                img = np.log(img)

                # STANDARDIZE PER IMAGE
                ep = 1e-6
                m = np.nanmean(img.flatten())
                s = np.nanstd(img.flatten())
                img = (img-m)/(s+ep)
                img = np.nan_to_num(img, nan=0.0)

                # CROP TO 256 TIME STEPS
                X[j,14:-14,:,k] = img[:,22:-22] / 2.0

            # EEG SPECTROGRAMS
            img = self.eeg_specs[row.eeg_id]
            X[j,:,:,4:] = img

            if self.mode!="test":
                y[j,] = row[self.TARGETS]

        return X,y