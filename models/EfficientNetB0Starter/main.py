VER = 5

# Contains all of the eegs together in one file - faster
# TODO: Local computers might not have enough ram, so consider one which processes as it goes.
TRAIN_METADATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
KAGGLE_SPECTROGRAM_DIR = "/kaggle/input/brain-spectrograms/specs.npy"
EEG_SPECTROGRAM_DIR = "/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy"

# # IF THIS EQUALS NONE, THEN WE TRAIN NEW MODELS
# # IF THIS EQUALS DISK PATH, THEN WE LOAD PREVIOUSLY TRAINED MODELS
LOAD_MODELS_FROM = "/kaggle/input/brain-efficientnet-models-v3-v4-v5/"
USE_KAGGLE_SPECTROGRAMS = True
USE_EEG_SPECTROGRAMS = True

# This gives the mapping between labels and their class id
LABEL_TO_ID = {"Seizure":0, "LPD":1, "GPD":2, "LRDA":3, "GRDA":4, "Other":5}
ID_TO_LABEL = {x:y for y,x in LABEL_TO_ID.items()}


# Initialise 2xT4 GPUs
import os, gc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
print("TensorFlow version =",tf.__version__)

# TODO: What to do about GPUs and Mixed Precision?
# # USE MULTIPLE GPUS
# gpus = tf.config.list_physical_devices("GPU")
# if len(gpus)<=1:
#     strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
#     print(f"Using {len(gpus)} GPU")
# else:
#     strategy = tf.distribute.MirroredStrategy()
#     print(f"Using {len(gpus)} GPUs")

# # USE MIXED PRECISION
# MIX = True
# if MIX:
#     tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
#     print("Mixed precision enabled")
# else:
#     print("Using full precision")


from input_utils import create_modified_eeg_metadata_df, PreloadedDataset
from constants import TARGETS

# 1. Reading Data --------------------------------------------------------------
# Ignore the sub_ids
eeg_metadata_df = pd.read_csv(TRAIN_METADATA_DIR)
eeg_metadata_df = create_modified_eeg_metadata_df(eeg_metadata_df)

# Read precomputed spectrograms (from kaggle and eeg)
# that are all placed in one dictionary
spectrograms = np.load(KAGGLE_SPECTROGRAM_DIR, allow_pickle=True).item()
all_eegs = np.load(EEG_SPECTROGRAM_DIR,allow_pickle=True).item()


# 2. Build Model ---------------------------------------------------------------

# Build EfficientNet Model

# !pip install --no-index --find-links=/kaggle/input/tf-efficientnet-whl-files /kaggle/input/tf-efficientnet-whl-files/efficientnet-1.1.1-py3-none-any.whl

import efficientnet.tfkeras as efn

def build_model():

    inp = tf.keras.Input(shape=(128,256,8))
    base_model = efn.EfficientNetB0(include_top=False, weights=None, input_shape=None)
    base_model.load_weights("/kaggle/input/tf-efficientnet-imagenet-weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5")

    # RESHAPE INPUT 128x256x8 => 512x512x3 MONOTONE IMAGE
    # KAGGLE SPECTROGRAMS
    x1 = [inp[:,:,:,i:i+1] for i in range(4)]
    x1 = tf.keras.layers.Concatenate(axis=1)(x1)
    # EEG SPECTROGRAMS
    x2 = [inp[:,:,:,i+4:i+5] for i in range(4)]
    x2 = tf.keras.layers.Concatenate(axis=1)(x2)
    # MAKE 512X512X3
    if USE_KAGGLE_SPECTROGRAMS & USE_EEG_SPECTROGRAMS:
        x = tf.keras.layers.Concatenate(axis=2)([x1,x2])
    elif USE_EEG_SPECTROGRAMS: x = x2
    else: x = x1
    x = tf.keras.layers.Concatenate(axis=3)([x,x,x])

    # OUTPUT
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(6,activation="softmax", dtype="float32")(x)

    # COMPILE MODEL
    model = tf.keras.Model(inputs=inp, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    loss = tf.keras.losses.KLDivergence()

    model.compile(loss=loss, optimizer = opt)

    return model


# 3. Train Model ---------------------------------------------------------------
## Train Model

from sklearn.model_selection import KFold, GroupKFold
import tensorflow.keras.backend as K, gc

all_oof = []
all_true = []

gkf = GroupKFold(n_splits=5)
for i, (train_index, valid_index) in enumerate(gkf.split(eeg_metadata_df, eeg_metadata_df.target, eeg_metadata_df.patient_id)):

    print("#"*25)
    print(f"### Fold {i+1}")

    # TODO: Pretty sure that PyTorch has some cross validation thing
    # So we only need one dataset.
    train_gen = PreloadedDataset(eeg_metadata_df.iloc[train_index], spectrograms, all_eegs,
                              shuffle=True, batch_size=32, augment=False)
    valid_gen = PreloadedDataset(eeg_metadata_df.iloc[valid_index], spectrograms, all_eegs,
                              shuffle=False, batch_size=64, mode="valid")

    print(f"### train size {len(train_index)}, valid size {len(valid_index)}")
    print("#"*25)

    K.clear_session()
    with strategy.scope():
        model = build_model()
    if LOAD_MODELS_FROM is None:
        model.fit(train_gen, verbose=1,
              validation_data = valid_gen,
              epochs=EPOCHS, callbacks = [LR])
        model.save_weights(f"EffNet_v{VER}_f{i}.h5")
    else:
        model.load_weights(f"{LOAD_MODELS_FROM}EffNet_v{VER}_f{i}.h5")

    oof = model.predict(valid_gen, verbose=1)
    all_oof.append(oof)
    all_true.append(eeg_metadata_df.iloc[valid_index][TARGETS].values)

    del model, oof
    gc.collect()

all_oof = np.concatenate(all_oof)
all_true = np.concatenate(all_true)


# Compute the CV score
import sys
sys.path.append("/kaggle/input/kaggle-kl-div")
from kaggle_kl_div import score

oof = pd.DataFrame(all_oof.copy())
oof["id"] = np.arange(len(oof))

true = pd.DataFrame(all_true.copy())
true["id"] = np.arange(len(true))

cv = score(solution=true, submission=oof, row_id_column_name="id")
print("CV Score KL-Div for EfficientNetB2 =",cv)



# 4. Test Model ---------------------------------------------------------------
# Infer test and crate submission
del all_eegs, spectrograms; gc.collect()
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape",test.shape)
test.head()


# READ ALL SPECTROGRAMS
PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
files2 = os.listdir(PATH2)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}
for i,f in enumerate(files2):
    if i%100==0: print(i,", ",end="")
    tmp = pd.read_parquet(f"{PATH2}{f}")
    name = int(f.split(".")[0])
    spectrograms2[name] = tmp.iloc[:,1:].values

# RENAME FOR DATALOADER
test = test.rename({"spectrogram_id":"spec_id"},axis=1)


import pywt, librosa

USE_WAVELET = None

NAMES = ["LL","LP","RP","RR"]

FEATS = [["Fp1","F7","T3","T5","O1"],
         ["Fp1","F3","C3","P3","O1"],
         ["Fp2","F8","T4","T6","O2"],
         ["Fp2","F4","C4","P4","O2"]]

# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet="haar", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])

    ret=pywt.waverec(coeff, wavelet, mode="per")

    return ret

def spectrogram_from_eeg(parquet_path, display=False):

    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128,256,4),dtype="float32")

    if display: plt.figure(figsize=(10,7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):

            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            # DENOISE
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256,
                  n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1]//32)*32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db+40)/40
            img[:,:,k] += mel_spec_db

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0

        if display:
            plt.subplot(2,2,k+1)
            plt.imshow(img[:,:,k],aspect="auto",origin="lower")
            plt.title(f"EEG {eeg_id} - Spectrogram {NAMES[k]}")

    if display:
        plt.show()
        plt.figure(figsize=(10,5))
        offset = 0
        for k in range(4):
            if k>0: offset -= signals[3-k].min()
            plt.plot(range(10_000),signals[k]+offset,label=NAMES[3-k])
            offset += signals[3-k].max()
        plt.legend()
        plt.title(f"EEG {eeg_id} Signals")
        plt.show()
        print(); print("#"*25); print()

    return img


# READ ALL EEG SPECTROGRAMS
PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
DISPLAY = 1
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("Converting Test EEG to Spectrograms..."); print()
for i,eeg_id in enumerate(EEG_IDS2):

    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f"{PATH2}{eeg_id}.parquet", i<DISPLAY)
    all_eegs2[eeg_id] = img


# INFER EFFICIENTNET ON TEST
preds = []
model = build_model()
#TODO: Should have a different DataGenerator for test
test_gen = PreloadedDataset(test, spectrograms2, all_eegs2,
                         shuffle=False, batch_size=64, mode="test")

for i in range(5):
    print(f"Fold {i+1}")
    if LOAD_MODELS_FROM:
        model.load_weights(f"{LOAD_MODELS_FROM}EffNet_v{VER}_f{i}.h5")
    else:
        model.load_weights(f"EffNet_v{VER}_f{i}.h5")
    pred = model.predict(test_gen, verbose=1)
    preds.append(pred)
pred = np.mean(preds,axis=0)
print()
print("Test preds shape",pred.shape)


sub = pd.DataFrame({"eeg_id":test.eeg_id.values})
sub[TARGETS] = pred
sub.to_csv("submission.csv",index=False)
print("Submissionn shape",sub.shape)
sub.head()


# SANITY CHECK TO CONFIRM PREDICTIONS SUM TO ONE
sub.iloc[:,-6:].sum(axis=1)