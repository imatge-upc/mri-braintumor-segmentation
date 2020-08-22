import os
from typing import Tuple

import torch
import numpy as np
from src.dataset.utils.nifi_volume import save_segmask_as_nifi_volume
from src.dataset.patient import Patient
from src.logging_conf import logger


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def predict(model, images: np.ndarray, device: torch.device, monte_carlo: bool=True) -> Tuple[np.ndarray, np.ndarray]:

    model.eval()
    if monte_carlo:
        enable_dropout(model)

    with torch.no_grad():

        images = torch.from_numpy(images).unsqueeze(0)
        inputs = images.float().to(device)

        four_channel_output, prediction_scores = model(inputs)

    return four_channel_output.detach().cpu(), prediction_scores.detach().cpu()


def get_prediction_map(four_channel_prediction: torch.tensor) -> np.ndarray:
    assert len(four_channel_prediction.shape) == 5
    return np.asarray(four_channel_prediction[0].max(0)[1].byte().cpu().data)

def get_scores_map_from_vector(vector_prediction_scores: np.ndarray, path_size: list) -> np.ndarray:
    assert len(vector_prediction_scores.shape) == 2, "Must be a 2d array with: (all_voxels, n_labels)"
    best_score, _ = vector_prediction_scores.max(1)
    return best_score.view(path_size)


def save_predictions(patient: Patient, results: dict, model_path: str, task: str):

    output_dir = os.path.join(model_path, task)
    output_dir_entropy = os.path.join(output_dir, "entropy")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir_entropy):
        os.makedirs(output_dir_entropy)

    for name, volume in results.items():
        file_name = f"{patient.patch_name}.nii.gz" if name == "prediction" else f"{patient.patch_name}_unc_{name}.nii.gz"

        directory = output_dir_entropy if "entropy" in file_name else output_dir
        output_path = os.path.join(directory, file_name)


        affine_func = patient.get_affine()
        logger.info(f"Saving to: {output_path}")
        save_segmask_as_nifi_volume(volume, affine_func, output_path)