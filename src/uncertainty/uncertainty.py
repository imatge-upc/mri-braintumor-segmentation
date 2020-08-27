from typing import Tuple, List
import torch
import numpy as np
from scipy import stats


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




