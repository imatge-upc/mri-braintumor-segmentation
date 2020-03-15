import logging
import numpy as np
import torch
from torch.utils.data import Dataset
import dataset.io as nifi_io
import dataset.nifi_volume_utils as nifi_utils

class BratsDataset(Dataset):

    flair_idx, t1_idx, t2_idx, t1_ce_idx = 0, 1, 2, 3

    def __init__(self, data, ground_truth, modalities_to_use, transform):
        super(BratsDataset, self).__init__()
        self.data = data
        self.gt = ground_truth
        self.transform = transform
        self.modalities_to_use = modalities_to_use


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        logging.debug(f'\nPACIENT: {idx}')
        flair = self._load_volume_modality(idx, BratsDataset.flair_idx)
        t1 = self._load_volume_modality(idx, BratsDataset.t1_idx)
        t2= self._load_volume_modality(idx, BratsDataset.t2_idx)
        t1_ce = self._load_volume_modality(idx, BratsDataset.t1_idx)
        segmentation_mask = self._load_volume_gt(idx)

        modalities = list(filter(lambda x: (x is not None), [flair, t1, t2, t1_ce]))

        paths = [self.data[idx, BratsDataset.flair_idx],
                 self.data[idx, BratsDataset.t1_idx],
                 self.data[idx, BratsDataset.t2_idx],
                 self.data[idx, BratsDataset.t1_ce_idx],
                 self.gt[idx]]

        modalities = np.asarray(modalities)
        segmentation_mask = np.asarray(segmentation_mask)
        return modalities, segmentation_mask, paths

    def _load_volume_gt(self, idx):
        return self._load_volume(nii_data=self.gt[idx])

    def _load_volume_modality(self, idx, modality):
        if modality in self.modalities_to_use.keys() and self.modalities_to_use[modality]:
            return self._load_volume(nii_data=self.data[idx, modality])
        else:
            return None

    def _load_volume(self, nii_data):
        logging.debug(nii_data)
        volume = nifi_io.load_nifi_volume(nii_data)
        volume = nifi_utils.resize_volume(volume, 0.3)
        return volume # self.transform(volume) if self.transform else volume