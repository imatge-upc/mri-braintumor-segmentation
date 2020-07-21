from typing import Tuple, List
import torch
import numpy as np

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

    return wt_var, tc_var, et_var