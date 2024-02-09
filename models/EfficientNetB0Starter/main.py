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
    # Doing a slice in order to keep the shape
    # (batch_size, frequency, time, spectrograms)
    # x1 is kaggle spectrograms and x2 is eeg spectrograms

    # Shape is 4 x(batch_size, 128, 256, 1)
    x1 = [inp[:,:,:,i:i+1] for i in range(4)]
    # Concatenate on axis 1 to get
    # (batch_size, 128 * 4, 256, 1) = (batch_size, 512, 256, 1)
    x1 = tf.keras.layers.Concatenate(axis=1)(x1)

    # EEG SPECTROGRAMS
    x2 = [inp[:,:,:,i+4:i+5] for i in range(4)]
    # Same, this gets (batch_size, 512, 256, 1)
    # (batch_size, 128 * 4, 256, 1) = (batch_size, 512, 256, 1)
    x2 = tf.keras.layers.Concatenate(axis=1)(x2)

    # MAKE 512X512X3
    if USE_KAGGLE_SPECTROGRAMS & USE_EEG_SPECTROGRAMS:
        # Put together to get (batch_size, 512, 256*2, 1)
        x = tf.keras.layers.Concatenate(axis=2)([x1,x2])
    elif USE_EEG_SPECTROGRAMS: x = x2
    else: x = x1

    # Same value for all the channels. I.e. gray scale image.
    # Here channel represents RGB instead.
    x = tf.keras.layers.Concatenate(axis=3)([x,x,x])

    # So you can think of the output as
    # ---> Axis 2
    # K1 E1
    # K2 E2
    # K3 E3
    # K4 E4
    # Vertical is axis 1, horizontal is axis 2.
    # Axis 3 is the colour, which is just gray scale as equal.
    # Summary: Input is just all the spectrograms tiled together into one
    # 512 x 512 image.
    # TODO: Is there a better way to write this down?

    # OUTPUT
    x = base_model(x)
    # Do pooling, form of convolution
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Have an MLP and softmax it
    x = tf.keras.layers.Dense(6,activation="softmax", dtype="float32")(x)

    # COMPILE MODEL
    model = tf.keras.Model(inputs=inp, outputs=x)
    # Using Adam with learning rate 1e-3
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    # Use KLD. TODO: Why does using cross entropy lead to a difference?
    loss = tf.keras.losses.KLDivergence()

    model.compile(loss=loss, optimizer = opt)

    return model


# 3. Train Model ---------------------------------------------------------------
## Train Model

from sklearn.model_selection import KFold, GroupKFold
import tensorflow.keras.backend as K, gc

# Output of the model
all_oof = []
# Actual output
all_true = []

# Grouping by patient id
# So for a given patient_id all instances of it will either be in the test or training set.
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
