import logging

import torch
from torch.utils.data import Dataset
from .io import load_nifi_volume

class BratsDataset(Dataset):

    def __init__(self, data, ground_truth, transform):
        super(BratsDataset, self).__init__()
        self.data = data
        self.gt = ground_truth
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        logging.debug(f'\nPACIENT: {idx}')
        logging.debug(self.data[idx, 0])
        logging.debug(self.data[idx, 1])
        logging.debug(self.data[idx, 2])
        logging.debug(self.data[idx, 3])

        flair = load_nifi_volume(self.data[idx, 0])
        t1 = load_nifi_volume(self.data[idx, 1])
        t2 = load_nifi_volume(self.data[idx, 2])
        t1_ce = load_nifi_volume(self.data[idx, 3])
        segmentation_mask = load_nifi_volume(self.gt[idx])

        if self.transform:
            flair = self.transform(flair)
            t1 = self.transform(t1)
            t2 = self.transform(t2)
            t1_ce = self.transform(t1_ce)
            segmentation_mask = self.transform(segmentation_mask)

        modalities = [flair, t1, t2, t1_ce]
        paths = [self.data[idx, 0], self.data[idx, 1], self.data[idx, 2], self.data[idx, 3], self.gt[idx]]
        return modalities, segmentation_mask, paths