def create_non_overlapping_eeg_data(df):
    """Only use one eeg for each eeg id as test dataset doesn"t have overlaps"""

    # Create non-overlapping eeg id data
    train = df.groupby("eeg_id")[["spectrogram_id","spectrogram_label_offset_seconds"]].agg(
        {"spectrogram_id":"first","spectrogram_label_offset_seconds":"min"})
    train.columns = ["spec_id","min"]

    tmp = df.groupby("eeg_id")[["spectrogram_id","spectrogram_label_offset_seconds"]].agg(
        {"spectrogram_label_offset_seconds":"max"})
    train["max"] = tmp

    tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
    train["patient_id"] = tmp

    tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
    for t in TARGETS:
        train[t] = tmp[t].values

    y_data = train[TARGETS].values
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train[TARGETS] = y_data

    tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
    train["target"] = tmp

    train = train.reset_index()
    print("Train non-overlapping eeg_id shape:", train.shape )
    train.head()

    return train


# Train Dataloader
class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"
    def __init__(self, data, batch_size=32, shuffle=False, augment=False, mode="train",
                 specs = spectrograms, eeg_specs = all_eegs):

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        # TODO: Not implemented yet, will need albumentations
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.on_epoch_end()

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
                y[j,] = row[TARGETS]

        return X,y