import os
from typing import List
import numpy as np
import csv
from src.dataset.patient import Patient

def read_brats(csv_path: str) -> List:
    patients = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        next(reader, None)
        for row in reader:
            patients.append(Patient(idx=row[0], center=row[3], grade=row[1], patient=row[2], patch_name=row[4],
                                    size=list(map(int, row[5].split("x"))), data_path=os.path.dirname(csv_path)))
    return patients


def create_roi_mask(data: np.ndarray) -> np.ndarray:
    # filter values bigger than 0
    data[data > 0.0] = 1
    data[data <= 0.0] = 0
    return data
