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


class KldClassifier(pl.LightningModule):
    """Takes a classifier which returns log probabilities
    Then uses KLD as the loss function.
    """
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def training_step(self, batch, batch_idx):
        x, y       = batch
        y_pred_log = self.classifier(x)
        # Compute the batch average of kld
        # kl_div assumes that y_pred_log is log_probabilities
        # And y unless specified are regular probabilities
        loss       = torch.nn.functional.kl_div(
            y_pred_log, y, reduction="batchmean")
        self.log("loss", loss)
        return loss

    def forward(self, x):
        """Return probabilities of classification"""
        return torch.exp(self.model(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict(self, dataloader):
        outputs = []
        # Don't need to compute gradients
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                output = self(batch)
                outputs.append(output)

        return torch.concatenate(outputs)


class TuchilusEfficientNetB0(torch.nn.Module):
    """Exploit the Starfish!

    Bias towards idealised and proto.
    TODO: Issue where we need to output log probabilities
    so potential for numerical errors here.
    """
    def __init__(self):
        super().__init__()
        # Have n + (n 2) options
        # 21 for our case as n = 6
        self.layers = torch.nn.Sequential(
            efficientnet_b0(num_classes=21),
            torch.nn.Softmax(dim=1)
        )
        self.tuchilus_matrix = self.generate_tuchilus_matrix()

    def forward(self, x):
        # (batch_size, 21)
        pre_probabilities = self.layers(x)
        # (batch_size, 6)
        probabilities = pre_probabilities @ self.tuchilus_matrix
        log_probabilities = torch.log(probabilities)
        return log_probabilities

    # TODO: Can we generalise this and make it more trainable?
    def generate_tuchilus_matrix(self):
        M = torch.nn.Parameter(torch.zeros(21, 6), requires_grad=False)
        for i in range(6):
            M[i, i] = 1

        # Going to be ordering these as
        # 1 2 3 4 5 6
        # 12 13 14 15 16
        # 23 24 25 26
        # 34 35 36
        # 45 46
        # 56
        # TODO: I can't quadratic formula
        jumps = [6, 5, 4, 3, 2, 1]
        for i in range(5):
            for j in range(i+1, 6):
                position = sum(jumps[:i+1]) + (j-i-1)
                M[position, i] = 0.5
                M[position, j] = 0.5

        return M
