import os

import numpy as np
import torch
from src.dataset import brats_labels
from torch.utils.data import Dataset
from torchvision import transforms

from src.dataset.utils import nifi_volume as nifi_utils
from src.logging_conf import logger


class BratsDataset(Dataset):

    flair_idx, t1_idx, t2_idx, t1ce_idx = 0, 1, 2, 3

    def __init__(self, data: list, modalities_to_use: dict, sampling_method,
                 patch_size: tuple, transforms: transforms):
        """

        :param data:
        :param ground_truth:
        :param modalities_to_use:
        :param sampling_method: patching method
        :param patch_size:
        :param transforms:
        """

        self.data = data
        self.modalities_to_use = modalities_to_use
        self.sampling_method = sampling_method
        self.patch_size = patch_size
        self.transforms = transforms
        self.normalize = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        root_path = os.path.join(self.data[idx].data_path, self.data[idx].patch_name)
        flair = self._load_volume_modality(os.path.join(root_path, self.data[idx].flair), BratsDataset.flair_idx)
        t1 = self._load_volume_modality(os.path.join(root_path, self.data[idx].t1), BratsDataset.t1_idx)
        t2 = self._load_volume_modality(os.path.join(root_path, self.data[idx].t2), BratsDataset.t2_idx)
        t1_ce = self._load_volume_modality(os.path.join(root_path, self.data[idx].t1ce), BratsDataset.t1_idx)
        modalities = np.asarray(list(filter(lambda x: (x is not None), [flair, t1, t2, t1_ce])))

        segmentation_mask = self._load_volume_gt(os.path.join(root_path, self.data[idx].seg))
        segmentation_mask = brats_labels.convert_from_brats_labels(segmentation_mask)

        # patch_modality, patch_segmentation = modalities, segmentation_mask
        patch_modality, patch_segmentation = self.sampling_method.patching(modalities, segmentation_mask, self.patch_size)
        return idx, patch_modality, patch_segmentation


    def _load_volume_modality(self, modality_path: str, modality: int):
        if modality in self.modalities_to_use.keys() and self.modalities_to_use[modality]:
            volume, _ = nifi_utils.load_nifi_volume(modality_path, self.normalize)
            return volume
        else:
            return None

    def _load_volume_gt(self, seg_mask: str) -> np.ndarray:
        segmentation, _ = nifi_utils.load_nifi_volume(seg_mask, normalize=False) # segmentation --> always false
        return segmentation

    def get_patient_info(self, idx):
        return {attr[0]: attr[1] for attr in vars(self.data[idx]).items()}
