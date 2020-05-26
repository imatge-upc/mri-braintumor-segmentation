import os
from typing import Tuple
import numpy as np
import nibabel as nib



def load_nifi_volume(filepath: str) -> np.ndarray:
    proxy = nib.load(filepath)
    img = proxy.get_fdata()
    proxy.uncache()
    return img


def get_dataset(rootdir: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.empty([0, 4], dtype='object')
    ground_truth = np.empty([0], dtype='object')
    _, dirs, _ = next(os.walk(rootdir))

    for tumor_type in dirs:
        tumor_type_path = os.path.join(rootdir, tumor_type)
        for _, dirs, _ in os.walk(tumor_type_path):
            for person in dirs:
                base_person_file = os.path.join(tumor_type_path, person, person)
                data = np.vstack((data, [
                    "{}_flair.nii.gz".format(base_person_file),
                    "{}_t1.nii.gz".format(base_person_file),
                    "{}_t2.nii.gz".format(base_person_file),
                    "{}_t1ce.nii.gz".format(base_person_file)
                ]))
                ground_truth = np.append(ground_truth, "{}_seg.nii.gz".format(base_person_file))

    return data, ground_truth


def get_dataset_path(local_path, server_path):

    if os.path.exists(local_path):
        root_path = local_path
    elif os.path.exists(server_path):
        root_path = server_path
    else:
        raise ValueError('No path is working')

    path_train = os.path.join(root_path, 'MICCAI_BraTS_2019_Data_Training/')
    path_test = os.path.join(root_path, 'MICCAI_BraTS_2019_Data_Validation/')

    return path_train, path_test