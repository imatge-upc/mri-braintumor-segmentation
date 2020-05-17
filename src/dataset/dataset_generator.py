import numpy as np
import torch
from torch.utils.data import Dataset
import dataset.io as nifi_io
import dataset.nifi_volume_utils as nifi_utils
from logging_conf import logger

class BratsDataset(Dataset):

    flair_idx, t1_idx, t2_idx, t1_ce_idx = 0, 1, 2, 3

    def __init__(self, data: np.ndarray, ground_truth: np.ndarray, modalities_to_use: dict, transform,  label:int=None):
        super(BratsDataset, self).__init__()
        self.data = data
        self.gt = ground_truth
        self.transform = transform
        self.modalities_to_use = modalities_to_use
        self.label = label


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

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
        if self.label:
            segmentation_mask = nifi_utils.get_one_label_volume(segmentation_mask, self.label)
        segmentation_mask = self._resize_volume(segmentation_mask)
        segmentation_mask = segmentation_mask[np.newaxis]

        return modalities, segmentation_mask, paths

    def _load_volume_gt(self, idx: int) -> np.ndarray:
        return self._load_volume(nii_data=self.gt[idx])

    def _load_volume_modality(self, idx: int, modality: int):
        if modality in self.modalities_to_use.keys() and self.modalities_to_use[modality]:
            volume =  self._load_volume(nii_data=self.data[idx, modality])
            return self._resize_volume(volume)
        else:
            return None

    def _load_volume(self, nii_data: str) -> np.ndarray:
        volume = nifi_io.load_nifi_volume(nii_data)
        return volume

    def _resize_volume(self, volume):
        return nifi_utils.resize_volume(volume)