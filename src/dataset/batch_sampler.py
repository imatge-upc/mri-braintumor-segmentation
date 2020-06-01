import importlib
from random import shuffle
from torch.utils.data import Sampler, DataLoader

from src.dataset.brats_dataset import BratsDataset


class BratsSampler(Sampler):

    def __init__(self, dataset, n_patients, n_samples):

        self.batch_size = n_patients*n_samples
        self.n_samples = n_samples
        self.n_patients = n_patients
        self.dataset_indices = list(range(0, len(dataset)))


    def __iter__(self):
        batch_indices = []
        shuffle(self.dataset_indices)

        for patient_id in self.dataset_indices:

            batch_indices.extend(patient_id for _ in range(self.n_samples))

            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []

        if len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        return (len(self.dataset_indices) + self.batch_size - 1) // self.batch_size



if __name__ == "__main__":
    p = "/Users/lauramora/Documents/MASTER/TFM/Data/"
    path_train, path_test = "MICCAI_BraTS_2019_Data_Training", "MICCAI_BraTS_2019_Data_Validation"

    modalities_to_use = {BratsDataset.flair_idx: True,
                         BratsDataset.t1_idx: True,
                         BratsDataset.t2_idx: True,
                         BratsDataset.t1ce_idx: True}

    sampling_method = importlib.import_module("src.dataset.patching.random_tumor_distribution")


    dataset = BratsDataset(path_train, modalities_to_use,sampling_method, (64,64,64))
    sampler = BratsSampler(dataset, 2, 2)
    train_loader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=1)

    for i, (volumes, segmentations) in enumerate(train_loader):
        print(f"item {i} {volumes.shape}")
        print(f"item {i} {segmentations.shape}")
