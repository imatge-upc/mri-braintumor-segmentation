import importlib
import os

from src.dataset.utils.visualization import plot_batch
from torch.utils.data import DataLoader

from src.dataset.utils.dataset import read_brats
from src.dataset.loaders.brats_dataset import BratsDataset
from tqdm import tqdm


def unnorm(data, epsilon=1e-8):
    non_zero = data[data > 0.0]
    mean = non_zero.mean()
    std = non_zero.std() + epsilon
    out = data*std + mean
    out[data == 0] = 0
    return out


dataset_path = "/Users/lauramora/Documents/MASTER/TFM/Data/2020/train/no_patch"
train_csv = os.path.join(dataset_path, "brats20_data.csv")

print("Loading dataset")
data, data_test = read_brats(train_csv)
data = data_test


sampling_method = importlib.import_module("src.dataset.patching.binary_distribution")


compute_patch =True
patch_size = (64,64,64)
batch_size = 4

print("Creating Dataset")
train_dataset = BratsDataset(data, sampling_method, patch_size, compute_patch=compute_patch)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

for data_batch, labels_batch in tqdm(train_loader, total=len(train_loader)):
    plot_batch(data_batch, seg=False)
    plot_batch(labels_batch, seg=True)


