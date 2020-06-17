import random
from typing import Tuple, List

import numpy as np
from src.dataset import dataset_utils


def add_patch(patches_by_patient: dict, index: int, patient: str):
    if patient not in patches_by_patient.keys():
        patches_by_patient[patient] = []
    patches_by_patient[patient].append(index)


def get_split_random(data: np.array, patches_by_patient: dict, val_size: float) -> Tuple[List, List]:
    patches_by_patient = list(patches_by_patient.values())
    val_n_elements = int(len(patches_by_patient) * val_size)

    validation_patient_indices = random.sample(range(0, len(patches_by_patient)), val_n_elements)

    train_patients, val_patients  = [], []
    for patient_index, sublist in enumerate(patches_by_patient):

        if patient_index in validation_patient_indices:
            val_patients.extend(data[patches_by_patient[patient_index]])
        else:
            train_patients.extend(data[patches_by_patient[patient_index]])

    return train_patients, val_patients



def train_val_split(data: list, val_size: float=0.25) -> Tuple[List, List]:
    patches_by_patient_lgg = {}
    patches_by_patient_hgg = {}

    data = np.array(data)

    for index, patient_patch in enumerate(data):
        patient = patient_patch.patient
        grade = patient_patch.grade

        if grade == "LGG":
            add_patch(patches_by_patient_lgg, index, patient)

        elif grade == "HGG":
            add_patch(patches_by_patient_hgg, index, patient)

        else:
            print("Unknown grade")

    train_patients_lgg, val_patients_lgg = get_split_random(data, patches_by_patient_lgg, val_size)
    train, val = get_split_random(data, patches_by_patient_hgg, val_size)

    train.extend(train_patients_lgg)
    val.extend(val_patients_lgg)

    return train, val


if __name__ == "__main__":
    csv_path = "/Users/lauramora/Documents/MASTER/TFM/Data/2020/train/random_tumor_distribution/brats20_data.csv"
    data = dataset_utils.read_brats(csv_path)
    train, val = train_val_split(data)
    print()