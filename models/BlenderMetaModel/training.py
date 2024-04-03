import numpy as np
import torch
import lightning as pl

from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import TensorBoardLogger


efficientnet_output = np.load("/home/delacruz-danilojr/delacruz/projects/SigSquad/models/BlenderMetaModel/efficientnet_evaluation_output.npy", allow_pickle=True).item()
wavenet_output = np.load("/home/delacruz-danilojr/delacruz/projects/SigSquad/models/BlenderMetaModel/wavenet_evaluation_output.npy", allow_pickle=True).item()


class Blender(torch.nn.Module):
    "Expecting log probabilities"
    def __init__(self, num_models=2):
        super().__init__()
        self.num_models = num_models
        self.num_classes = 6

        # 16 as it's next power of 2 after 12
        self.hidden_layer_size = 32
        self.layers = torch.nn.Sequential(
            # 16 as it's next power of 2 after 12
            torch.nn.Linear(self.num_classes * self.num_models, self.hidden_layer_size, bias=True),
            torch.nn.ReLU(),

            torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size, bias=True),
            torch.nn.ReLU(),

            torch.nn.Linear(self.hidden_layer_size, self.num_classes, bias=True),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, stacked_log_prob):
        return self.layers(stacked_log_prob)


class KldClassifier(pl.LightningModule):
    """Takes a classifier which returns log probabilities
    Then uses KLD as the loss function.
    """
    def __init__(self, classifier, learning_rates=[1e-3]):
        super().__init__()
        self.classifier = classifier
        self.learning_rates = learning_rates

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
        return self.optimizer

    def on_train_epoch_start(self):
        # TODO: Maybe can use one of the schedulers?
        # Update learning rate at the beginning of each epoch
        # Epoch does not change over different trainers
        epoch = self.current_epoch
        # Do not change the learning rate when we run out of new ones
        if epoch < len(self.learning_rates):
            lr = self.learning_rates[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr


class TrainDataset(Dataset):

    def __init__(self):
        eff_oof = torch.concatenate(efficientnet_output["all_oof"], axis=0)
        wave_oof = torch.concatenate(wavenet_output["all_oof"], axis=0)

        self.X = torch.concatenate([eff_oof, wave_oof], axis=1)

        # Should be the same among efficientnet and wavenet
        self.y = torch.concatenate(efficientnet_output["all_true"], axis=0)

    @staticmethod
    def transform(x):
        return torch.log(x)

    def __getitem__(self, idx):
        X = self.transform(self.X[idx, :])
        y = self.y[idx, :]
        return X, y

    def __len__(self):
        return self.X.shape[0]


blender = Blender()
model = KldClassifier(blender, learning_rates=[
        1e-3, 1e-3, 1e-3,
        1e-3, 1e-3, 1e-3,
        1e-3, 1e-3, 1e-3,
        1e-4, 1e-4, 1e-4,
        1e-5, 1e-5, 1e-5,
    ])

logger = TensorBoardLogger("tb_logs")
trainer = pl.Trainer(max_epochs=5*3, log_every_n_steps=1)
train_dataset = TrainDataset()
train_loader = DataLoader(train_dataset, batch_size=8, num_workers=15)
trainer.fit(model=model, train_dataloaders=train_loader)


MODEL_DIR = "blend_meta"

torch.save(model.state_dict(), f"{MODEL_DIR}/{i}.pt")