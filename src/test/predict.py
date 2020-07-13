import os
from typing import Tuple

import torch
import numpy as np
from src.dataset import brats_labels
from src.dataset.utils.nifi_volume import save_segmask_as_nifi_volume
from src.dataset.patient import Patient
from src.logging_conf import logger


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def predict(model, patient: Patient, add_padding: bool, device: torch.device, monte_carlo: bool=True,
            save: bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    if monte_carlo:
        enable_dropout(model)

    with torch.no_grad():
        images = patient.load_mri_volumes()

        if add_padding:
            new_array = np.zeros((4, 240, 240, 240))
            new_array[:,  :images.shape[1], :images.shape[2], :images.shape[3]] = images
            images = new_array

        images = torch.from_numpy(images).unsqueeze(0)
        inputs = images.float().to(device)

        prediction, prediction_scores = model(inputs)
        prediction = np.asarray(prediction[0].max(0)[1].byte().cpu().data)

        best_score, best_prediction_map_vector = prediction_scores.max(1) # get best prediction

        prediction_map = np.asarray(best_prediction_map_vector.view(images[0][0].shape).cpu().data)
        best_scores_map = np.asarray(best_score.view(images[0][0].shape).cpu().data)

        prediction_map = brats_labels.convert_to_brats_labels(prediction_map)

    output_path = os.path.join(patient.data_path, patient.patch_name, f"{patient.patch_name}_prediction.nii.gz")
    flair_path = os.path.join(patient.data_path, patient.patch_name, patient.flair)
    if add_padding:
        prediction_map = prediction_map[:, :, :155]

    if save:
        logger.info(f"Saving prediction to: {output_path}")
        save_segmask_as_nifi_volume(prediction_map, flair_path, output_path)

    return prediction_scores, prediction_map, best_scores_map, prediction

