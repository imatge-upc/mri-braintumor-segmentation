import numpy as np


def filter_by_threshold_eval_regions(T: int, prediction: np.ndarray, wt_unc: np.ndarray, tc_unc: np.ndarray, et_unc: np.ndarray) -> np.ndarray:
    pred = prediction.copy()
    pred[et_unc > T] = 1
    pred[tc_unc > T] = 2
    pred[wt_unc > T] = 0
    return pred
