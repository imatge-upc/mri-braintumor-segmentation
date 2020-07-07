from typing import Tuple
from medpy import metric
import numpy as np


def get_confusion_matrix(prediction: np.ndarray, reference: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Computes tp/fp/tn/fn from teh provided segmentations
    """
    assert prediction.shape == reference.shape, "'prediction' and 'reference' must have the same shape"

    tp = int(((prediction != 0) * (reference != 0)).sum()) # overlap
    fp = int(((prediction != 0) * (reference == 0)).sum())
    tn = int(((prediction == 0) * (reference == 0)).sum()) # no segmentation
    fn = int(((prediction == 0) * (reference != 0)).sum())

    return tp, fp, tn, fn


def dice(tp: int, fp:int, fn:int) -> float:
    """
    Dice coefficient computed using the definition of true positive (TP), false positive (FP), and false negative (FN)
    2TP / (2TP + FP + FN)
    """
    denominator = 2*tp + fp + fn
    if denominator <= 0:
        return 0
    return (2 * tp / denominator)

# Hausdorff
def hausdorff(prediction: np.ndarray, reference: np.ndarray) -> float:
    try:
        return metric.hd95(prediction, reference)
    except:
        print(f"Prediction does not contain the same label as gt. "
              f"Pred labels {np.unique(prediction)} GT labels {np.unique(reference)}")
        return None


# Sensitivity: recall
def recall(tp, fn) -> float:
    """TP / (TP + FN)"""
    actual_positives = tp + fn
    if actual_positives <= 0:
        return 0
    return tp / actual_positives

# Specificity: precision
def precision(tp, fp) -> float:
    """TP/ (TP + FP)"""
    predicted_positives = tp + fp
    if predicted_positives <= 0:
        return 0
    return tp / predicted_positives


def fscore(tp, fp, tn, fn, beta:int=1) -> float:
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""
    assert beta > 0

    precision_ = precision(tn, fp)
    recall_ = recall(tp, fn)

    if ((beta * beta * precision_) + recall_) <= 0:
        return 0

    fscore = (1 + beta * beta) * precision_ * recall_ / ((beta * beta * precision_) + recall_)
    return fscore


def accuracy(tp, fp, tn, fn) -> float:
    """(TP + TN) / (TP + FP + FN + TN)"""
    if (tp + fp + tn + fn) <= 0:
        return 0
    return (tp + tn) / (tp + fp + tn + fn)


