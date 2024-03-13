# Contains all of the eegs together in one file - faster
TRAIN_METADATA_DIR     = "../../train.csv"
KAGGLE_SPECTROGRAM_DIR = "../../train_spectrograms/"
EEG_SPECTROGRAM_DIR    = "../../EEG_Spectrograms"


import pandas as pd
import torch
import lightning as pl

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger

from input_utils import modify_train_metadata, TrainDataset, LazySpectrogramFetcher
from model import KldClassifier, EfficientNetB0Starter
from constants import TARGETS

# 1. Obtain Metadata and Loaders -----------------------------------------------

train_metadata = pd.read_csv(TRAIN_METADATA_DIR)
train_metadata = modify_train_metadata(train_metadata)

train_fetcher = LazySpectrogramFetcher(KAGGLE_SPECTROGRAM_DIR, EEG_SPECTROGRAM_DIR)
train_dataset = TrainDataset(train_metadata, train_fetcher)

torch.set_float32_matmul_precision('medium')
logger = TensorBoardLogger("tb_logs")
train_loader = DataLoader(train_dataset, batch_size=8, num_workers=15)
trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1)

# 2. Train Model and save weights ----------------------------------------------

enb0 = EfficientNetB0Starter()
model = KldClassifier(enb0)
trainer.fit(model=model, train_dataloaders=train_loader)

torch.save(model.state_dict(), "trained_model_weights.pt")
