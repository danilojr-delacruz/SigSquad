import torch
import os
import numpy as np
import pandas as pd
from utils import modify_metadata, TARGETS
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, metadata, features):
        self.metadata = metadata
        self.features = self.preprocess_features(features)
        self.labels = torch.Tensor(metadata[TARGETS].values)

    def preprocess_features(self, features):
        # fill na
        features = torch.nan_to_num(features)
        # normalize features
        self.mean = torch.mean(features, axis=[0,1])
        self.std = torch.std(features, axis=[0,1])
        features = (features - self.mean) / (self.std + 1e-6)


        features = torch.clamp(features, -3, 3)

        return features.to(torch.float32)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.features[idx]
        label = self.labels[idx]
        return sample, label
    
class KaggleTestDataset(Dataset):
    """Labels are not available"""
    def __init__(self, metadata, features, mean, std):
        self.metadata = metadata
        self.mean = mean
        self.std = std
        self.features = self.preprocess_features(features, mean, std)


    def preprocess_features(self, features, mean, std):
        # fill na
        features = torch.nan_to_num(features)
        # normalize features
        features = (features - mean) / (std + 1e-6)
        features = torch.clamp(features, -3, 3)

        return features.to(torch.float32)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.features[idx]
        return sample

