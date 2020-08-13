from typing import Tuple, List
import torch
import numpy as np
from tqdm import tqdm

from src.test import predict


def get_variation_uncertainty(prediction_score_vectors: List[torch.tensor], matrix_size: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute uncertainty using the variance in the predictions for the evaluation metrics WT, TC, ET
    :param prediction_score_vectors: list of tensors containing the predicted scores x each label computed using TTD
    :return:
    """
    prediction_score_vectors = torch.stack(tuple(prediction_score_vectors))

    wt_var = np.var(np.sum(prediction_score_vectors[:, :, 1:].cpu().numpy(), axis=2), axis=0).reshape(matrix_size) * 100
    tc_var = np.var(np.sum(prediction_score_vectors[:, :, [1, 3]].cpu().numpy(), axis=2), axis=0).reshape( matrix_size) * 100
    et_var = np.var(prediction_score_vectors[:, :, 3].cpu().numpy(), axis=0).reshape(matrix_size) * 100

    return wt_var.astype(np.uint8), tc_var.astype(np.uint8), et_var.astype(np.uint8)


def ttd_uncertainty_loop(model, images, device, K=2):
    prediction_labels_maps, prediction_score_vectors = [], []

    for _ in tqdm(range(K), desc="Predicting.."):
        prediction_four_channels, vector_prediction_scores = predict.predict(model, images,
                                                                             device, monte_carlo=True)

        prediction_labels_maps.append(predict.get_prediction_map(prediction_four_channels))
        prediction_score_vectors.append(vector_prediction_scores)

    return prediction_labels_maps, prediction_score_vectors


def brats_normalize(uncertainty_map: np.ndarray, max_unc: int, min_unc: int) -> np.ndarray:
    minimum = 0
    maximum = 100

    step = (maximum - minimum) / (max_unc - min_unc)

    vfunc = np.vectorize(lambda x: (x - min_unc) * step if x != 0 else 0)

    return vfunc(uncertainty_map).astype(np.uint8)




