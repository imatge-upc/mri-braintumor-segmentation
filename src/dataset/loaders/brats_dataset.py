import os

import numpy as np
from src.dataset import brats_labels
from torch.utils.data import Dataset
from torchvision import transforms

from src.dataset.utils import nifi_volume as nifi_utils
import torch



class BratsDataset(Dataset):

    flair_idx, t1_idx, t2_idx, t1ce_idx = 0, 1, 2, 3

    def __init__(self, data: list, sampling_method, patch_size: tuple, transforms: transforms, compute_patch: bool=False):
        """

        :param data:
        :param ground_truth:
        :param modalities_to_use:
        :param sampling_method: patching method
        :param patch_size:
        :param transforms:
        """
        self.data = data
        self.sampling_method = sampling_method
        self.patch_size = patch_size
        self.transforms = transforms
        self.compute_patch = compute_patch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        root_path = os.path.join(self.data[idx].data_path, self.data[idx].patch_name)
        flair = self._load_volume_modality(os.path.join(root_path, self.data[idx].flair))
        t1 = self._load_volume_modality(os.path.join(root_path, self.data[idx].t1))
        t2 = self._load_volume_modality(os.path.join(root_path, self.data[idx].t2))
        t1_ce = self._load_volume_modality(os.path.join(root_path, self.data[idx].t1ce))
        modalities = np.asarray(list(filter(lambda x: (x is not None), [flair, t1, t2, t1_ce])))

        segmentation_mask = self._load_volume_gt(os.path.join(root_path, self.data[idx].seg))
        segmentation_mask = brats_labels.convert_from_brats_labels(segmentation_mask)

        if self.compute_patch:
            patch_modality, patch_segmentation = self.sampling_method.patching(modalities, segmentation_mask, self.patch_size)
        else:
            patch_modality, patch_segmentation = modalities, segmentation_mask

        patch_modality = torch.from_numpy(patch_modality.astype(float))
        patch_segmentation = torch.from_numpy(patch_segmentation.astype(int))

        return idx, patch_modality, patch_segmentation


    def _load_volume_modality(self, modality_path: str):
        volume = nifi_utils.load_nifi_volume(modality_path, True)
        return  volume

    def _load_volume_gt(self, seg_mask: str) -> np.ndarray:
        segmentation = nifi_utils.load_nifi_volume(seg_mask, normalize=False)
        return segmentation

    def get_patient_info(self, idx):
        return {attr[0]: attr[1] for attr in vars(self.data[idx]).items()}
