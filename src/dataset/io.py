import os
from typing import Tuple
import numpy as np
import nibabel as nib


def load_nifi_volume(filepath: str):
    proxy = nib.load(filepath)
    img = proxy.get_fdata()
    proxy.uncache()
    return img


def get_dataset(rootdir: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.empty([0, 4], dtype='object')
    ground_truth = np.empty([0], dtype='object')
    _, dirs, _ = next(os.walk(rootdir))

    for tumor_type in dirs:
        for _, dirs, _ in os.walk(f"{rootdir}{tumor_type}"):
            for person in dirs:
                base_person_file = f"{rootdir}{tumor_type}/{person}/{person}"
                data = np.vstack((data, [
                    f"{base_person_file}_flair.nii.gz",
                    f"{base_person_file}_t1.nii.gz",
                    f"{base_person_file}_t2.nii.gz",
                    f"{base_person_file}_t1ce.nii.gz"
                ]))
                ground_truth = np.append(ground_truth, f"{base_person_file}_seg.nii.gz")

    return data, ground_truth


