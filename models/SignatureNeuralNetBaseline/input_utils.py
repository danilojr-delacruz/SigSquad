import torch
import os
import numpy as np
import pandas as pd
from utils import modify_metadata, TARGETS
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    """Assumes precomputed signatures are available"""
    def __init__(self, metadata, features):
        self.metadata = metadata
        self.features = self.preprocess_features(features)
        self.labels = torch.Tensor(metadata[TARGETS].values)

    def preprocess_features(self, features):
        # fill na
        features = torch.nan_to_num(features)
        # normalize features
        features = (features - torch.mean(features, axis=0)) / (torch.std(features, axis=0) + 1e-6)
        features = torch.clamp(features, -3, 3)

        return features.to(torch.float32)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.features[idx]
        label = self.labels[idx]
        return sample, label