from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
import lightning as pl
from torchvision.models import efficientnet_b0


class EfficientNetB0Starter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Under the hood this will
        # Take in (batch_size, 3, 512, 512) image
        # Return convolution features of shape
        # (batch_size, 1280, 16, 16)
        # Perform Global Max Average Pooling and Flattent
        # (batch_size, 1280)
        # Perform dropout (B0 uses p=0.2)
        # Then a linear layer to get shape
        # (batch_size, num_classes)
        # Then apply softmax to get probabilities
        self.layers = torch.nn.Sequential(
            efficientnet_b0(num_classes=6),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)


class LitENB0(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = EfficientNetB0Starter()

    def training_step(self, batch, batch_idx):
        x, y       = batch
        y_pred_log = self.model(x)
        # Compute the batch average of kld
        # kl_div assumes that y_pred_log is log_probabilities
        # And y unless specified are regular probabilities
        loss       = torch.nn.functional.kl_div(
            y_pred_log, y, reduction="batchmean")
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

