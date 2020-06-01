from typing import Tuple
import os
import numpy as np

from src.dataset.visualization_utils import plot_3_view
from src.dataset.nifi_volume_utils import load_nifi_volume


dataset_path = "/Users/lauramora/Documents/MASTER/TFM/Data/MICCAI_BraTS_2019_Data_Training/HGG/"
patient = "BraTS19_CBICA_AAB_1"
save = False

def plot(volume: np.ndarray, patch: np.ndarray, volume_slice: int = 100):
    plot_3_view("flair", volume[0, :, :, :], volume_slice, save=True)
    plot_3_view("patch_flair", patch[0, :, :, :], volume_slice, save=True)


def load_patient() -> Tuple[np.ndarray, np.ndarray]:
    flair = load_nifi_volume(os.path.join(dataset_path, patient, f"{patient}_flair.nii.gz"))
    t1 = load_nifi_volume(os.path.join(dataset_path, patient, f"{patient}_t1.nii.gz"))
    t1ce = load_nifi_volume(os.path.join(dataset_path, patient, f"{patient}_t1ce.nii.gz"))
    t2 = load_nifi_volume(os.path.join(dataset_path, patient, f"{patient}_t2.nii.gz"))
    masks = load_nifi_volume(os.path.join(dataset_path, patient, f"{patient}_seg.nii.gz"))

    modalities = np.asarray([t1, t1ce, t2, flair])

    return modalities, masks


def patching_strategy(patching_method, size: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    volume, seg = load_patient()
    volume_patches, seg_patches = patching_method(volume, seg, size)
    return volume, volume_patches, seg_patches