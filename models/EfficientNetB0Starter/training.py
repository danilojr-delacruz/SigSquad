# Contains all of the eegs together in one file - faster
TRAIN_METADATA_DIR     = "../../train.csv"
KAGGLE_SPECTROGRAM_DIR = "../../train_spectrograms/"
EEG_SPECTROGRAM_DIR    = "../../EEG_Spectrograms"

MODEL_DIR = "experts_only"

import os
import pandas as pd
import torch
import lightning as pl

from torch.utils.data import DataLoader, Subset
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import GroupKFold

from input_utils import modify_train_metadata, TrainDataset, ValidationDataset, \
    LazySpectrogramFetcher
from model import KldClassifier, EfficientNetB0Starter
from constants import TARGETS

NUM_FOLDS = 5

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Obtain Metadata and Loaders -----------------------------------------------

train_metadata = pd.read_csv(TRAIN_METADATA_DIR)
train_metadata = modify_train_metadata(train_metadata)

# Measure how non uniform a sample is.
# TODO: Add this in modify_train_metadata once confirmed
kld_with_uniform = torch.nn.functional.kl_div(
    torch.log(torch.tensor((train_metadata.loc[:, TARGETS] + 1e-5).values)),
    torch.ones(6) / 6,
    reduction="none").sum(dim=-1, keepdim=True)
train_metadata["Non-uniformity"] = kld_with_uniform

# This returns a boolean indexer
hard_index = train_metadata["Non-uniformity"] < 5.5
# Want a numerical indexer
hard_index = hard_index[hard_index].index

train_fetcher = LazySpectrogramFetcher(KAGGLE_SPECTROGRAM_DIR, EEG_SPECTROGRAM_DIR)
train_dataset = TrainDataset(train_metadata, train_fetcher)

val_dataset = ValidationDataset(train_metadata, train_fetcher)

torch.set_float32_matmul_precision('medium')
logger = TensorBoardLogger("tb_logs")

# 2. Train Model and save weights ----------------------------------------------

# Output of the model
all_oof = []
# Actual output
all_true = []
# CV Scores
oof_cv = []

# Grouping by patient id
# So for a given patient_id all instances of it will either be in the test or training set.
gkf = GroupKFold(n_splits=NUM_FOLDS)
for i, (train_index, valid_index) in enumerate(
    gkf.split(train_metadata, train_metadata.target, train_metadata.patient_id)):

    print(f"Fold {i}")
    # Build model
    enb0 = EfficientNetB0Starter()
    # Restricting to only good samples, went down from 17k to 6k
    # So triple each epoch round so same number of training steps
    # The hard index stays the same though at 3177 so lets just keep it
    # hard round the same
    model = KldClassifier(enb0, learning_rates=[
        1e-3, 1e-3, 1e-3,
        1e-3, 1e-3, 1e-3,
        1e-3, 1e-3, 1e-3,
        1e-4, 1e-4, 1e-4,
        1e-5, 1e-5, 1e-5,
        # Hard round
        1e-5, 1e-5, 1e-6
    ])

    # Train model
    print("Training on all data")
    trainer = pl.Trainer(max_epochs=5*3, log_every_n_steps=1)
    train_loader = DataLoader(Subset(train_dataset, train_index),
                              batch_size=8, num_workers=15)
    trainer.fit(model=model, train_dataloaders=train_loader)

    print("Training on non-uniform samples")
    trainer.fit_loop.max_epochs += 3
    hard_train_loader = DataLoader(Subset(
        train_dataset, list(set(train_index) & set(hard_index))),
                              batch_size=8, num_workers=15)
    trainer.fit(model=model, train_dataloaders=hard_train_loader)

    # Save weights
    torch.save(model.state_dict(), f"{MODEL_DIR}/{i}.pt")

    val_loader = DataLoader(Subset(val_dataset, valid_index),
                            batch_size=8, num_workers=15, shuffle=False)

    # Compute Out of Fold Score
    model.eval()
    with torch.no_grad():
        # This should be a tensor (b, 6)
        oof_pred = torch.concat(trainer.predict(model, val_loader))

    # These should match as val_loader has shuffle=False
    oof_true = torch.tensor(train_metadata.loc[valid_index, TARGETS].values)

    all_oof.append(oof_pred)
    all_true.append(oof_true)

# 3. Compute CV Scores ----------------------------------------------

# Compute CV Score by fold
for i in range(NUM_FOLDS):
    # TODO: Get the log probabilities directly
    oof_cv.append(torch.nn.functional.kl_div(
            torch.log(all_oof[i]), all_true[i], reduction="batchmean"))

# Compute the Overall CV score
oof = torch.concat(all_oof)
true = torch.concat(all_true)

overall_oof_cv = torch.nn.functional.kl_div(
    torch.log(oof), true, reduction="batchmean"
)

with open(f"{MODEL_DIR}/cv_scores.txt", "w") as f:
    for i, cv_score in enumerate(oof_cv):
        f.write(f"Fold {i}, CV = {cv_score:.3f}\n")
    f.write(f"Overall CV Score = {overall_oof_cv:.3f}")
