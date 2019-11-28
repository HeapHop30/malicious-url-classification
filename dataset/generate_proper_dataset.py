import numpy as np
import pandas as pd

np.random.seed(42)

def binary_undersample(dataset):
    " Reduces number of negative samples to make it more similar to number of positive ones "
    bigger_class = 0
    smaller_class = 1
    epsilon = 0.07
    ratio_to_reach = round(len(dataset[dataset.label == smaller_class]) / len(dataset), 2) + epsilon
    print(ratio_to_reach)
    filter_mask = np.random.choice(a=[True, False], size=len(dataset[dataset.label == bigger_class]), p=[ratio_to_reach, 1-ratio_to_reach])
    class_samples = dataset[dataset.label == bigger_class]
    filtered_class_samples = class_samples[filter_mask]
    new_dataset = pd.concat((filtered_class_samples, dataset[dataset.label == smaller_class]))
    return new_dataset

PATH_TO_DATA = "data.csv"
NEW_PATH_TO_DATA = "preprocessed_data.csv"

dataset = pd.read_csv(PATH_TO_DATA)

print(f"Dataset shape: {dataset.shape}")
print(f"Labels: {dataset.label.unique()}")

n_bad = dataset[dataset.label == 'bad'].shape[0]
n_good = dataset[dataset.label == 'good'].shape[0]
print(f"Bad: {n_bad}\t\t({n_bad*100/len(dataset):.0f}%)")
print(f"Good: {n_good}\t({n_good*100/len(dataset):.0f}%)")

dataset.label[dataset.label == 'bad'] = 1
dataset.label[dataset.label == 'good'] = 0
print(f"New labels: {dataset.label.unique()}")

dataset = binary_undersample(dataset)

dataset.to_csv(NEW_PATH_TO_DATA, header=['url', 'label'], index=False)

