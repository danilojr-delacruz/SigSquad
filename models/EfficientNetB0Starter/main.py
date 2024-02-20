# Contains all of the eegs together in one file - faster
TRAIN_METADATA_DIR     = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
KAGGLE_SPECTROGRAM_DIR = "/kaggle/input/brain-spectrograms/specs.npy"
EEG_SPECTROGRAM_DIR    = "/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy"

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


from input_utils import modify_train_metadata, PreloadedSpectrogramFetcher, \
                        TrainDataset, TestDataset, \
                        preload_spectrograms, preload_eeg_spectrograms
from model import LitENB0
from constants import TARGETS
from torch.utils.data import DataLoader

import torch

# 1. Reading Data --------------------------------------------------------------
# Ignore the sub_ids
train_metadata = pd.read_csv(TRAIN_METADATA_DIR)
train_metadata = modify_train_metadata(train_metadata)

# Read precomputed spectrograms (from kaggle and eeg)
# that are all placed in one dictionary
spectrograms     = np.load(KAGGLE_SPECTROGRAM_DIR, allow_pickle=True).item()
eeg_spectrograms = np.load(EEG_SPECTROGRAM_DIR,allow_pickle=True).item()

# 2. Build and Train Model -----------------------------------------------------

model = LitENB0()
model.load_state_dict(torch.load("../../personal_sandbox/trained_model_19-02-2024.pt"))

del eeg_spectrograms, spectrograms; gc.collect()
# 3. Test Model ---------------------------------------------------------------

test_metadata = pd.read_csv(TEST_METADATA_DIR)

# READ ALL SPECTROGRAMS
test_spectrograms     = preload_spectrograms(TEST_SPECTROGRAM_DIR)
# TODO: In training, was the spectrogram eeg over the entire duration or just the desired segment
# It was precomputed for us so we don't know.
# If so, essentially left hand side is full spectrogram, and right hand side is zoomed in
test_eeg_spectrograms = preload_eeg_spectrograms(test_metadata, TEST_EEG_DIR)

# READ ALL EEG SPECTROGRAMS AND CONVERT

fetcher      = PreloadedSpectrogramFetcher(test_spectrograms, test_eeg_spectrograms)
test_dataset = TestDataset(test_metadata, fetcher, shuffle=False)
test_loader  = DataLoader(test_dataset)

predictions = model.predict(test_loader)

# Ensure predictions sum to 1
predictions /= predictions.sum(dim=1)

submission = pd.DataFrame({"eeg_id": test_metadata.eeg_id.values})
submission[TARGETS] = predictions
submission.to_csv("submission.csv", index=False)
