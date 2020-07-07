import importlib
import os
import sys
import csv
from src.dataset.augmentations.brats_augmentations import zero_mean_unit_variance_normalization
from src.dataset import nifi_volume_utils as nifi_utils
from src.dataset.nifi_volume_utils import save_nifi_volume

from src.dataset import dataset_utils
from src.config import BratsConfiguration
from tqdm import tqdm
import numpy as np

def load_volume(volume_path: str, normalize: bool=True) -> np.ndarray:
    volume = nifi_utils.load_nifi_volume(volume_path)
    if normalize:
        volume = zero_mean_unit_variance_normalization(volume)
    return volume


if __name__ == "__main__":
    config = BratsConfiguration(sys.argv[1])
    model_config = config.get_model_config()
    dataset_config = config.get_dataset_config()
    basic_config = config.get_basic_config()

    patch_size = config.patch_size
    patch_size_str = "x".join(list(map(str, patch_size)))
    n_patches = dataset_config.getint("n_patches")

    data_csv = "/mnt/gpid07/users/laura.mora/datasets/2020/train/no_patch/brats20_data.csv"
    data, data_test  = dataset_utils.read_brats(data_csv)
    data.extend(data_test)

    sampling_method = importlib.import_module(dataset_config.get("sampling_method"))
    sampling_name = dataset_config.get("sampling_method").split(".")[-1]
    method_path = f"{dataset_config.get('root_path')}/train/{sampling_name}"
    if not os.path.exists(method_path):
        os.makedirs(method_path)

    with open(f"{method_path}/brats20_data.csv", "w") as file:

        writer = csv.writer(file)
        writer.writerow(["ID","Grade","subject_ID", "Center", "Patch", "Size", "Train"])
        idx = 0
        for patient in tqdm(data, total=len(data), desc="Computing patches"):
            patient_path = os.path.join(patient.data_path, patient.patch_name)
            flair = load_volume(os.path.join(patient_path, patient.flair), True)
            t1 =    load_volume(os.path.join(patient_path, patient.t1), True)
            t2 =    load_volume(os.path.join(patient_path, patient.t2), True)
            t1_ce = load_volume(os.path.join(patient_path, patient.t1ce), True)
            seg = load_volume(os.path.join(patient_path, patient.seg), False)

            modalities = np.asarray(list(filter(lambda x: (x is not None), [flair, t1, t2, t1_ce])))

            for patch in range(0, n_patches):

                patient_patch_name = f"{patient.patient}_p{patch}_{patch_size_str}"
                patch_path = f"{method_path}/{patient_patch_name}"
                if not os.path.exists(patch_path):
                    os.makedirs(patch_path)

                print(f"Path: {patch_path}")
                patch_modality, patch_segmentation = sampling_method.patching(modalities, seg, patch_size)

                seg_path = f"{patch_path}/{patient_patch_name}_seg.nii.gz"
                flair_path = f"{patch_path}/{patient_patch_name}_flair.nii.gz"
                t1_path = f"{patch_path}/{patient_patch_name}_t1.nii.gz"
                t2_path = f"{patch_path}/{patient_patch_name}_t2.nii.gz"
                t1ce_path = f"{patch_path}/{patient_patch_name}_t1ce.nii.gz"

                save_nifi_volume(patch_segmentation, seg_path)
                save_nifi_volume(patch_modality[0], flair_path)
                save_nifi_volume(patch_modality[1], t1_path)
                save_nifi_volume(patch_modality[2], t2_path)
                save_nifi_volume(patch_modality[3], t1ce_path)

                train_or_test = "train" if patient.train else "test"
                writer.writerow([idx, patient.grade, patient.patch_name, patient.center, patient_patch_name,
                                 patch_size_str, train_or_test])
                idx += 1