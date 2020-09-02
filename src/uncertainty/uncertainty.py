from typing import Tuple, List
import torch
import numpy as np
from scipy import stats
import os
from tqdm import tqdm

from src.dataset.utils import nifi_volume


def get_entropy_uncertainty(prediction_score_vectors: List[torch.tensor], matrix_size: Tuple) -> np.ndarray:
    """
    Compute uncertainty using the entropy of the predictions in the predictions for the evaluation metrics WT, TC, ET
    :param prediction_score_vectors: list of tensors containing the predicted scores x each label computed using TTD
    :return: a prediction map with the global uncertainty value
    """
    prediction_score_vectors = torch.stack(tuple(prediction_score_vectors))
    mean = np.mean(prediction_score_vectors.cpu().numpy(), axis=0)
    entropy = stats.entropy(mean, axis=1).reshape(matrix_size) * 100
    return entropy.astype(np.uint8)


def get_variation_uncertainty(prediction_score_vectors: List[torch.tensor], matrix_size: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute uncertainty using the variance in the predictions for the evaluation metrics WT, TC, ET
    :param prediction_score_vectors: list of tensors containing the predicted scores x each label computed using TTD
    :return:
    """
    prediction_score_vectors = torch.stack(tuple(prediction_score_vectors))

    wt_var = np.var(np.sum(prediction_score_vectors[:, :, 1:].cpu().numpy(), axis=2), axis=0).reshape(matrix_size) * 100
    tc_var = np.var(np.sum(prediction_score_vectors[:, :, [1, 3]].cpu().numpy(), axis=2), axis=0).reshape( matrix_size) *100
    et_var = np.var(prediction_score_vectors[:, :, 3].cpu().numpy(), axis=0).reshape(matrix_size) * 100

    return wt_var.astype(np.uint8), tc_var.astype(np.uint8), et_var.astype(np.uint8)


def brats_normalize(uncertainty_map: np.ndarray, max_unc: int, min_unc: int) -> np.ndarray:
    minimum = 0
    maximum = 100
    step = (maximum - minimum) / (max_unc - min_unc)
    vfunc = np.vectorize(lambda x: (x - min_unc) * step if x != 0 else 0)
    return vfunc(uncertainty_map).astype(np.uint8)


def compute_normalization(input_dir, output_dir, ground_truth_path):

    file_list = sorted([file for file in os.listdir(input_dir) if "unc" in file and "nii.gz"])
    file_list_all = sorted([file for file in os.listdir(input_dir) if "nii.gz" in file])

    max_uncertainty = 0
    min_uncertainty = 10000

    for uncertainty_map in tqdm(file_list, total=len(file_list), desc="Getting min and max"):

        # Load Uncertainty maps
        patient_name = uncertainty_map.split(".")[0].split("_unc")[0]
        path_gt = os.path.join(ground_truth_path, patient_name, f"{patient_name}_flair.nii.gz")
        flair = nifi_volume.load_nifi_volume(path_gt, normalize=False)
        brain_mask = np.zeros(flair.shape, np.float)
        brain_mask[flair > 0] = 1

        path = os.path.join(input_dir, uncertainty_map)
        unc_map, _ = nifi_volume.load_nifi_volume_return_nib(path, normalize=False)

        tmp_max = np.max(unc_map[brain_mask == 1])
        tmp_min = np.min(unc_map[brain_mask == 1])

        if tmp_max > max_uncertainty:
            max_uncertainty = tmp_max

        if tmp_min < min_uncertainty:
            min_uncertainty = tmp_min

    for uncertainty_map_path in tqdm(file_list_all, total=len(file_list_all), desc="Normalizing.."):

        path = os.path.join(input_dir, uncertainty_map_path)
        output_path = os.path.join(output_dir, uncertainty_map_path)

        unc_map, nib_data = nifi_volume.load_nifi_volume_return_nib(path, normalize=False)

        if "unc" in uncertainty_map_path:
            uncertainty_map_normalized = brats_normalize(unc_map, max_unc=max_uncertainty, min_unc=min_uncertainty)
            print(f"Saving to: {output_path}")
            nifi_volume.save_segmask_as_nifi_volume(uncertainty_map_normalized, nib_data.affine, output_path)
        else:
            nifi_volume.save_segmask_as_nifi_volume(unc_map, nib_data.affine, output_path)