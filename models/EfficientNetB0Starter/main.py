VER = 5

# Contains all of the eegs together in one file - faster
# TODO: Local computers might not have enough ram, so consider one which processes as it goes.
TRAIN_METADATA_DIR     = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
KAGGLE_SPECTROGRAM_DIR = "/kaggle/input/brain-spectrograms/specs.npy"
EEG_SPECTROGRAM_DIR    = "/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy"

# # IF THIS EQUALS NONE, THEN WE TRAIN NEW MODELS
# # IF THIS EQUALS DISK PATH, THEN WE LOAD PREVIOUSLY TRAINED MODELS
LOAD_MODELS_FROM = "/kaggle/input/brain-efficientnet-models-v3-v4-v5/"
USE_KAGGLE_SPECTROGRAMS = True
USE_EEG_SPECTROGRAMS = True


TEST_METADATA_DIR    = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
TEST_SPECTROGRAM_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms"
TEST_EEG_DIR         = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"


# Initialise 2xT4 GPUs
import os, gc
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import pandas as pd
import numpy as np

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


from input_utils import modify_train_metadata, PreloadedSpectrogramFetcher, TrainDataset, TestDataset
from model import LitENB0
from utils import spectrogram_from_eeg
from constants import TARGETS
from torch.utils.data import DataLoader

import torch

# 1. Reading Data --------------------------------------------------------------
# Ignore the sub_ids
train_metadata = pd.read_csv(TRAIN_METADATA_DIR)
train_metadata = modify_train_metadata(train_metadata)

# Read precomputed spectrograms (from kaggle and eeg)
# that are all placed in one dictionary
spectrograms = np.load(KAGGLE_SPECTROGRAM_DIR, allow_pickle=True).item()
all_eegs = np.load(EEG_SPECTROGRAM_DIR,allow_pickle=True).item()

# 2. Build and Train Model -----------------------------------------------------

model = LitENB0()
model.load_state_dict(torch.load("../../personal_sandbox/trained_model_19-02-2024.pt"))

# 3. Test Model ---------------------------------------------------------------
del all_eegs, spectrograms; gc.collect()

test_metadata = pd.read_csv(TEST_METADATA_DIR)

# READ ALL SPECTROGRAMS
test_spectrograms = {}
for filename in os.listdir(TEST_SPECTROGRAM_DIR):
    spectrogram = pd.read_parquet(f"{TEST_SPECTROGRAM_DIR}/{filename}")
    # Filenames look like 12345.parquet
    name        = int(filename.split(".")[0])
    # Remove the first column which represents time
    test_spectrograms[name] = spectrogram.iloc[:, 1:].values

# READ ALL EEG SPECTROGRAMS AND CONVERT
test_all_eegs = {}
for i, eeg_id in enumerate(test_metadata.eeg_id.unique()):
    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f"{TEST_EEG_DIR}/{eeg_id}.parquet")
    test_all_eegs[eeg_id] = img

fetcher      = PreloadedSpectrogramFetcher(test_spectrograms, test_all_eegs)
test_dataset = TestDataset(test_metadata, shuffle=False)
test_loader  = DataLoader(test_dataset)

predictions = model.predict(test_loader)

# Ensure predictions sum to 1
predictions /= predictions.sum(dim=1)

submission = pd.DataFrame({"eeg_id": test_metadata.eeg_id.values})
submission[TARGETS] = predictions
submission.to_csv("submission.csv", index=False)
