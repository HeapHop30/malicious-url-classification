import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class MaliciousURL(Dataset):
    def __init__(self, data):
        self.X, self.y = data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = str(self.X[idx])
        label = int(self.y[idx])

        return sample, label


def import_dataset(path):
    dataset = pd.read_csv(path)
    dataset = dataset.to_numpy()
    print(f"Dataset shape: {dataset.shape}")
    return dataset


def split_dataset(dataset):
    test_split = 0.15
    val_split = 0.4
    X_train, X_test, y_train, y_test = train_test_split(dataset[:,0], dataset[:,1], test_size = test_split, random_state = 42, shuffle=True)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_split, random_state=42, shuffle=True)

    train_set = MaliciousURL((X_train, y_train))
    test_set = MaliciousURL((X_test, y_test))
    val_set = MaliciousURL((X_val, y_val))

    return train_set, test_set, val_set


def get_dataloaders(sets, batch_size=32):
    train_set, test_set, val_set = sets
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader
