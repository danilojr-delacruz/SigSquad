from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import numpy as np
import torch
import lightning as pl

from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint


efficientnet_output = np.load("/home/delacruz-danilojr/delacruz/projects/SigSquad/models/BlenderMetaModel/efficientnet_evaluation_output.npy", allow_pickle=True).item()
wavenet_output = np.load("/home/delacruz-danilojr/delacruz/projects/SigSquad/models/BlenderMetaModel/wavenet_evaluation_output.npy", allow_pickle=True).item()
resnet_output = np.load("/home/delacruz-danilojr/delacruz/projects/SigSquad/models/BlenderMetaModel/sig_resnet_evaluation_output.npy", allow_pickle=True).item()


# Ensure these are tensors
for i in range(5):
    resnet_output["all_oof"][i] = torch.tensor(resnet_output["all_oof"][i])

torch.manual_seed(2962848)


activation = "ReLU"
hidden_layer_width = 32

activation_factory = {
    "ReLU": torch.nn.ReLU,
    "Tanh": torch.nn.Tanh,
}[activation]


class Blender(torch.nn.Module):
    "Expecting log probabilities"
    def __init__(self, num_models=3, hidden_layer_size=hidden_layer_width):
        super().__init__()
        self.num_models = num_models
        self.num_classes = 6

        # 16 as it's next power of 2 after 12
        self.hidden_layer_size = hidden_layer_size
        self.layers = torch.nn.Sequential(
            # 16 as it's next power of 2 after 12
            torch.nn.Linear(self.num_classes * self.num_models, self.hidden_layer_size, bias=True),
            activation_factory(),

            torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size, bias=True),
            activation_factory(),

            # torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size, bias=True),
            # activation_factory(),

            torch.nn.Linear(self.hidden_layer_size, self.num_classes, bias=True),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, stacked_log_prob):
        return self.layers(stacked_log_prob)


class KldClassifier(pl.LightningModule):
    """Takes a classifier which returns log probabilities
    Then uses KLD as the loss function.
    """
    def __init__(self, classifier):
        super().__init__()
        self.save_hyperparameters()
        self.classifier = classifier

    def training_step(self, batch, batch_idx):
        x, y       = batch
        y_pred_log = self.classifier(x)
        # Compute the batch average of kld
        # kl_div assumes that y_pred_log is log_probabilities
        # And y unless specified are regular probabilities
        loss       = torch.nn.functional.kl_div(
            y_pred_log, y, reduction="batchmean")
        self.log("loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Assume that no y is passed in
        x = batch
        return torch.exp(self.classifier(x))

    def forward(self, x):
        """Return probabilities of classification"""
        return torch.exp(self.classifier(x))

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 40000)

        return [self.optimizer], [self.scheduler]

    # def on_train_epoch_start(self):
    #     # TODO: Maybe can use one of the schedulers?
    #     # Update learning rate at the beginning of each epoch
    #     # Epoch does not change over different trainers
    #     epoch = self.current_epoch
    #     # Do not change the learning rate when we run out of new ones
    #     if epoch < len(self.learning_rates):
    #         lr = self.learning_rates[epoch]
    #         for param_group in self.optimizer.param_groups:
    #             param_group['lr'] = lr

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred_log = self.classifier(x)
        loss       = torch.nn.functional.kl_div(
            y_pred_log, y, reduction="batchmean")
        self.log("val_loss", loss, prog_bar=True)


# eff_oof = torch.concatenate(efficientnet_output["all_oof"][:-1], axis=0)
# wave_oof = torch.concatenate(wavenet_output["all_oof"][:-1], axis=0)
# res_oof = torch.concatenate(resnet_output["all_oof"][:-1], axis=0)
eff_oof = torch.concatenate(efficientnet_output["all_oof"], axis=0)
wave_oof = torch.concatenate(wavenet_output["all_oof"], axis=0)
res_oof = torch.concatenate(resnet_output["all_oof"], axis=0)
X_train = torch.concatenate([eff_oof, wave_oof, res_oof], axis=1)
# Should be the same among efficientnet and wavenet
# y_train = torch.concatenate(efficientnet_output["all_true"][:-1], axis=0)
y_train = torch.concatenate(efficientnet_output["all_true"], axis=0)

# Prevent leakeage by working with patient folds
eff_oof = efficientnet_output["all_oof"][-1]
wave_oof = wavenet_output["all_oof"][-1]
res_oof = resnet_output["all_oof"][-1]
X_val = torch.concatenate([eff_oof, wave_oof, res_oof], axis=1)
# Should be the same among efficientnet and wavenet
y_val = efficientnet_output["all_true"][-1]

class TrainDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    @staticmethod
    def transform(x):
        return torch.log(x)

    def __getitem__(self, idx):
        X = self.transform(self.X[idx, :])
        y = self.y[idx, :]
        return X, y

    def __len__(self):
        return self.X.shape[0]


class EarlyStoppingWithWarmup(EarlyStopping):
    """
    EarlyStopping, except don't watch the first `warmup` epochs.
    https://github.com/Lightning-AI/pytorch-lightning/issues/2644
    min_epochs doesn't work for me...
    """
    def __init__(self, warmup=10, **kwargs):
        super().__init__(**kwargs)
        self.warmup = warmup

    def on_validation_end(self, trainer, pl_module):
        return self.on_epoch_end(trainer, pl_module)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup:
            return
        else:
            super()._run_early_stopping_check(trainer, pl_module)


blender = Blender()
model = KldClassifier(blender)
# model = KldClassifier.load_from_checkpoint("/home/delacruz-danilojr/delacruz/projects/SigSquad/models/BlenderMetaModel/ReLU_hlw=16_other_seed/model-epoch=34-val_loss=0.4263.ckpt")

config_name = f"three_models_{activation}_hlw={hidden_layer_width}_two_flat_layer"
checkpoint_path = f"/home/delacruz-danilojr/delacruz/projects/SigSquad/models/BlenderMetaModel/{config_name}"

import os; os.makedirs(checkpoint_path, exist_ok=True)
import shutil; shutil.rmtree(checkpoint_path) # Clear directory
os.makedirs(checkpoint_path, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=checkpoint_path,  # directory to save the checkpoints
    filename="model-{epoch:02d}-{val_loss:.4f}",  # filename template
    save_top_k=3,  # save the top 3 models based on the monitored quantity
    mode="min",  # mode for comparison: 'min' or 'max'
    )

logger = TensorBoardLogger("tb_logs")
trainer = pl.Trainer(max_steps=6000,
                     log_every_n_steps=1, accelerator="cuda",
                    #  callbacks=[checkpoint_callback]
                    #  callbacks=[EarlyStopping(
                    #      monitor="val_loss", mode="min", patience=5)]
                     )

train_dataset = TrainDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=3)

# val_dataset = TrainDataset(X_val, y_val)
# val_loader = DataLoader(val_dataset, batch_size=32, num_workers=3)

trainer.fit(model=model,
            train_dataloaders=train_loader,
            # val_dataloaders=val_loader,
            )

torch.save(model.state_dict(), "trained_all_data_20_epochs_more_layers.pt")

