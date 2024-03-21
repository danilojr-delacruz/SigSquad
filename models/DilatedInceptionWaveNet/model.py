import math
import torch
import lightning as pl


class GatedTCN(torch.nn.Module):
    """Gated temporal convolution layer.

    Parameters:
        conv_module: customized convolution module
    """

    def __init__(self, in_dim, h_dim, kernel_size, dilation_factor,
        dropout=None, conv_module=None):
        super().__init__()

        # Model blocks
        if conv_module is None:
            conv_module = torch.nn.Conv2d
            kernel_size = (1, kernel_size)

        self.filter = conv_module(
            in_channels=in_dim, out_channels=h_dim,
            kernel_size=kernel_size, dilation=dilation_factor
        )
        self.gate = conv_module(
            in_channels=in_dim, out_channels=h_dim,
            kernel_size=kernel_size, dilation=dilation_factor
        )

        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        """Forward pass.

        Parameters:
            x: input sequence

        Return:
            h: output sequence

        Shape:
            x: (B, C, N, L), where L denotes the input sequence length
            h: (B, h_dim, N, L')
        """
        x_filter = torch.nn.functional.tanh(self.filter(x))
        x_gate = torch.nn.functional.sigmoid(self.gate(x))
        h = x_filter * x_gate
        if self.dropout is not None:
            h = self.dropout(h)

        return h


class WaveBlock(torch.nn.Module):
    """WaveNet block.

    Args:
        kernel_size: kernel size, pass a list of kernel sizes for
            inception
    """

    def __init__(self, n_layers, in_dim, h_dim, kernel_size, conv_module=None):
        super().__init__()

        self.n_layers = n_layers
        self.dilation_rates = [2**l for l in range(n_layers)]

        self.in_conv = torch.nn.Conv2d(in_dim, h_dim, kernel_size=(1, 1))
        self.gated_tcns = torch.nn.ModuleList()
        self.skip_convs = torch.nn.ModuleList()
        for layer in range(n_layers):
            c_in, c_out = h_dim, h_dim
            self.gated_tcns.append(
                GatedTCN(
                    in_dim=c_in,
                    h_dim=c_out,
                    kernel_size=kernel_size,
                    dilation_factor=self.dilation_rates[layer],
                    conv_module=conv_module,
                )
            )
            self.skip_convs.append(torch.nn.Conv2d(
                h_dim, h_dim, kernel_size=(1, 1)))

        # Initialize parameters
        torch.nn.init.xavier_uniform_(self.in_conv.weight,
                                      gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.zeros_(self.in_conv.bias)
        for i in range(len(self.skip_convs)):
            torch.nn.init.xavier_uniform_(self.skip_convs[i].weight,
                                          gain=torch.nn.init.calculate_gain("relu"))
            torch.nn.init.zeros_(self.skip_convs[i].bias)

    def forward(self, x):
        """Forward pass.

        Shape:
            x: (B, C, N, L), where C denotes in_dim
            x_skip: (B, C', N, L), where C' denotes h_dim
        """
        # Input convolution
        x = self.in_conv(x)

        x_skip = x
        for layer in range(self.n_layers):
            x = self.gated_tcns[layer](x)
            x = self.skip_convs[layer](x)

            # Skip-connection
            x_skip = x_skip + x

        return x_skip


class DilatedInception(torch.nn.Module):
    """Dilated inception layer.

    Note that `out_channels` will be split across #kernels.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        # Network parameters
        n_kernels = len(kernel_size)
        assert out_channels % n_kernels == 0, "`out_channels` must be divisible by #kernels."
        h_dim = out_channels // n_kernels

        # Model blocks
        self.convs = torch.nn.ModuleList()
        for k in kernel_size:
            self.convs.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=h_dim,
                    kernel_size=(1, k),
                    padding="same",
                    dilation=dilation),
            )

    def forward(self, x):
        """Forward pass.

        Parameters:
            x: input sequence

        Return:
            h: output sequence

        Shape:
            x: (B, C, N, L), where C = in_channels
            h: (B, C', N, L'), where C' = out_channels
        """
        x_convs = []
        for conv in self.convs:
            x_conv = conv(x)
            x_convs.append(x_conv)
        h = torch.cat(x_convs, dim=1)

        return h


class DilatedInceptionWaveNet(torch.nn.Module):
    """WaveNet architecture with dilated inception conv."""

    def __init__(self):
        super().__init__()

        kernel_size = [2, 3, 6, 7]

        # Model blocks
        self.wave_module = torch.nn.Sequential(
            WaveBlock(12, 1, 16, kernel_size, DilatedInception),
            WaveBlock(8, 16, 32, kernel_size, DilatedInception),
            WaveBlock(4, 32, 64, kernel_size, DilatedInception),
            WaveBlock(1, 64, 64, kernel_size, DilatedInception),
        )
        # This returns a log probability
        self.output = torch.nn.Sequential(
            torch.nn.Linear(64 * 4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 6),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        """Forward pass.

        Shape:
            x: (B, L, C)
        """
        bs, length, in_dim = x.shape
        x = x.transpose(1, 2).unsqueeze(dim=2)  # (B, C, N, L), N is redundant

        x_ll_1 = self.wave_module(x[:, 0:1, :])
        x_ll_2 = self.wave_module(x[:, 1:2, :])
        x_ll = (torch.nn.functional.adaptive_avg_pool2d(x_ll_1, (1, 1))
                + torch.nn.functional.adaptive_avg_pool2d(x_ll_2, (1, 1))) / 2

        x_rl_1 = self.wave_module(x[:, 2:3, :])
        x_rl_2 = self.wave_module(x[:, 3:4, :])
        x_rl = (torch.nn.functional.adaptive_avg_pool2d(x_rl_1, (1, 1))
                + torch.nn.functional.adaptive_avg_pool2d(x_rl_2, (1, 1))) / 2

        x_lp_1 = self.wave_module(x[:, 4:5, :])
        x_lp_2 = self.wave_module(x[:, 5:6, :])
        x_lp = (torch.nn.functional.adaptive_avg_pool2d(x_lp_1, (1, 1))
                + torch.nn.functional.adaptive_avg_pool2d(x_lp_2, (1, 1))) / 2

        x_rp_1 = self.wave_module(x[:, 6:7, :])
        x_rp_2 = self.wave_module(x[:, 7:8, :])
        x_rp = (torch.nn.functional.adaptive_avg_pool2d(x_rp_1, (1, 1))
                + torch.nn.functional.adaptive_avg_pool2d(x_rp_2, (1, 1))) / 2

        x = torch.cat([x_ll, x_rl, x_lp, x_rp], axis=1).reshape(bs, -1)
        output = self.output(x)

        return output


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


class MeanAggregator(torch.nn.Module):
    def __init__(self, classifiers):
        super().__init__()
        self.num_classifiers = len(classifiers)
        # TODO: Making these children nodes by setting as attributes
        for i in range(self.num_classifiers):
            setattr(self, f"classifier_{i}", classifiers[i])

    def forward(self, x):
        predictions_log = [getattr(self, f"classifier_{i}")(x)
                           for i in range(self.num_classifiers)]
        # TODO: Using math.log to avoid working doing
        # torch.log(torch.tensor([self.num_classifiers]))
        # Issue is that this will always be on CPU
        mean_predictions_log = torch.logsumexp(
            torch.stack(predictions_log), dim=0) - \
            math.log(self.num_classifiers)
        return mean_predictions_log
