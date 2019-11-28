import torch
import torch.nn as nn
import data
import models
# from utils import train, test

# Global variables
PATH_TO_DATA = "dataset/preprocessed_data.csv"
random_seed = 42
on_gpu = False

# Setting the seed
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    on_gpu = True

# Parameters
params = {
    'epochs': 10,
    'batch_size': 32
}

# Load data
dataset = data.import_dataset(PATH_TO_DATA)
train_set, test_set, val_set = data.split_dataset(dataset)
train_loader, test_loader, val_loader = data.get_dataloaders((train_set, test_set, val_set), batch_size=params['batch_size'])
loaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
    }



