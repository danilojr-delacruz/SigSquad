import torch
    
class FlatModel(torch.nn.Module):
    def __init__(self, sig_dimension, dropout, hidden_layer_dim):
        super(FlatModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(sig_dimension, 6),
            torch.nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        return self.model(x)