import torch
from model import EnsembleModel
import os


def train(model, train_loader, test_loader, device, criterion, optimizer, early_stopping_epochs, max_epochs=100):
    train_losses = []
    test_losses = []

    model.to(device)
    for epoch in range(max_epochs):
        train_loss = 0
        model.train()
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(torch.log(y_pred), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(torch.log(y_pred), y)
            test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        # early stopping
        if epoch > early_stopping_epochs and test_losses[-1] > test_losses[-2]:
            break

        # if the loss is nan, break
        if test_losses[-1] != test_losses[-1]:
            print('Loss is nan')
            break

    return train_losses, test_losses, model

def CV_score(dataset, lr, weight_decay, dropout, classifier_input_dim, hidden_layer_dim, device, criterion, early_stopping_epochs, folds=5, save_models=False):
    """Train and evaluate the model for each data fold.
    """
    fold_size = len(dataset) // folds
    scores = []
    train_losses = []
    test_losses = []
    for fold in range(folds):
        train_dataset = torch.utils.data.Subset(dataset, list(range(0, fold*fold_size)) + list(range((fold+1)*fold_size, len(dataset))))
        test_dataset = torch.utils.data.Subset(dataset, list(range(fold*fold_size, (fold+1)*fold_size)))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=fold_size, shuffle=False)
        sig_dimension = dataset[0][0].shape[1]
        ensemble_model = EnsembleModel(sig_dimension, dropout, classifier_input_dim, hidden_layer_dim)
        optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=lr, weight_decay=weight_decay)
        train_loss, test_loss, model = train(ensemble_model, train_loader, test_loader, device, criterion, optimizer, early_stopping_epochs, max_epochs=100)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if save_models:
            # save the model parameters
            folder_name = f"{lr}_{weight_decay}_{dropout}_{classifier_input_dim}_{hidden_layer_dim}"
            if not os.path.exists(f"model_logs/{folder_name}"):
                os.makedirs(f"model_logs/{folder_name}")
            torch.save(model.state_dict(), f"model_logs/{folder_name}/model_{fold}.pt")
        if test_loss[-1] != test_loss[-1]:
            scores.append(test_loss[-2])
        else:
            scores.append(test_loss[-1])
    return scores, train_losses, test_losses