import torch

class EnsembleModel(torch.nn.Module):
    def __init__(self, sig_dimension, dropout, classifier_input_dim, hidden_layer_dim):
        super(EnsembleModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(sig_dimension, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_layer_dim, classifier_input_dim),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(classifier_input_dim*4, classifier_input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(classifier_input_dim, classifier_input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(classifier_input_dim, 6),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # put each brain region through the model, then concatenate and put into classifier
        outputs = []
        for i in range(4):
            outputs.append(self.model(x[:, i, :]))
        return self.classifier(torch.cat(outputs, axis=1))
    
class FlatModel(torch.nn.Module):
    def __init__(self, sig_dimension, dropout, classifier_input_dim, hidden_layer_dim):
        super(FlatModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(sig_dimension*4, hidden_layer_dim),
            torch.nn.BatchNorm1d(hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.BatchNorm1d(hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_layer_dim, 6),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = x.reshape(-1, x.shape[1]*x.shape[2])
        return self.model(x)