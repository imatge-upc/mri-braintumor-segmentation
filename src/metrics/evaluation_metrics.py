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
    return (2 * tp / denominator)

# Hausdorff
def hausdorff(prediction: np.ndarray, reference: np.ndarray) -> float:
    return metric.hd95(prediction, reference)

# Sensitivity: recall
def sensitivity(tp, fn) -> float:
    """TP / (TP + FN)"""
    all_positives = tp + fn
    return tp / all_positives

# Specificity: precision
def specificity(tn, fp) -> float:
    """TN / (TN + FP)"""
    all_negatives = tn + fp
    return tn / all_negatives


def fscore(tp, fp, tn, fn, beta:int=1) -> float:
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""
    assert beta > 0

    precision = specificity(tn, fp)
    recall = sensitivity(tp, fn)

    fscore = (1 + beta * beta) * precision * recall / ((beta * beta * precision) + recall)
    return fscore


def accuracy(tp, fp, tn, fn) -> float:
    """(TP + TN) / (TP + FP + FN + TN)"""
    return (tp + tn) / (tp + fp + tn + fn)


