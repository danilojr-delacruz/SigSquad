# Contains all of the eegs together in one file - faster
TEST_METADATA_DIR    = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
TEST_SPECTROGRAM_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms"
TEST_EEG_DIR         = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"

MODEL_DIR            = "/kaggle/input/efficientnetb0-19-02-2024/trained_model_19-02-2024.pt"


import pandas as pd
import torch
import lightning as pl

from torch.utils.data import DataLoader

from input_utils import LazyEegFetcher, TestDataset
from model import KldClassifier, DilatedInceptionWaveNet
from constants import TARGETS

# 1. Build and Load Model ------------------------------------------------------
wavenet = DilatedInceptionWaveNet()
model = KldClassifier(wavenet)
model.load_state_dict(torch.load(MODEL_DIR))

# 2. Generate Predictions ------------------------------------------------------

# Put into evaluation mode, turns off dropout
model.eval()
test_metadata = pd.read_csv(TEST_METADATA_DIR)

fetcher      = LazyEegFetcher(TEST_EEG_DIR)
test_dataset = TestDataset(test_metadata, fetcher, shuffle=False)
# TODO: Can speed up evaluations with GPUs later
test_loader  = DataLoader(test_dataset, num_workers=3)

trainer = pl.Trainer()
predictions = torch.concat(trainer.predict(model, test_loader))

# 3. Create Submission ---------------------------------------------------------

# Ensure predictions sum to 1
predictions /= predictions.sum(dim=1).unsqueeze(1)

submission = pd.DataFrame({"eeg_id": test_metadata.eeg_id.values})
submission[TARGETS] = predictions
submission.to_csv("submission.csv", index=False)
