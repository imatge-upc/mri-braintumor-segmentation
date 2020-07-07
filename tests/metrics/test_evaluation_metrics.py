import pytest
import numpy as np
from src.metrics import evaluation_metrics as metrics

@pytest.fixture(scope="function")
def volume():
    return np.random.randint(4, size=(4, 4, 4))


def test_confusion_matrix_correct_output(volume):
    tp, fp, tn, fn = metrics.get_confusion_matrix(volume, volume)

    non_zero = np.count_nonzero(volume)
    zero = np.prod(volume.shape) - non_zero

    assert tp == non_zero
    assert tn == zero
    assert fp == 0
    assert fn == 0

def test_confusion_matrix_assertation_error_size_mismatch(volume):
    with pytest.raises(Exception) as e:
        ref = np.random.randint(4, size=(2, 2, 4))
        _, _, _, _ = metrics.get_confusion_matrix(volume, ref)


def test_hausdorff_distance_correct_output(volume):
    hd = metrics.hausdorff(volume, volume)
    assert hd == 0

def test_perfect_dice_score():
    dice_score = metrics.dice(tp=75, fp=0, fn=0)
    assert dice_score == 1

def test_50_percent_dice_score():
    dice_score = metrics.dice(tp=30, fp=40, fn=20)
    assert dice_score == 0.5

def test_dice_zerodivicion():
    assert metrics.dice(tp=0, fp=0, fn=0) == 0

def test_precision_correct():
    precision = metrics.precision(tp=10, fp=0)
    assert precision == 1

def test_precision_50_percent():
    precision = metrics.precision(tp=10, fp=10)
    assert precision == 0.5

def test_precision_zerodivicion():
    precision = metrics.precision(tp=0, fp=0)
    assert precision == 0

def test_recall_correct():
    recall = metrics.recall(tp=10, fn=0)
    assert recall == 1

def test_recall_50_percent():
    recall = metrics.recall(tp=10, fn=10)
    assert recall == 0.5

def test_recall_zerodivicion():
    assert metrics.recall(tp=0, fn=0) == 0

def test_acc_correct():
    recall = metrics.accuracy(tp=10, fp=0, tn=10, fn=0)
    assert recall == 1

def test_acc_50_percent():
    recall = metrics.accuracy(tp=10, fp=10, tn=10, fn=10)
    assert recall == 0.5

def test_acc_zerodivicion():
    assert metrics.accuracy(tp=0, fp=0, tn=0, fn=0) == 0

def test_fscore_correct():
    recall = metrics.fscore(tp=10, fp=0, tn=10, fn=0)
    assert recall == 1

def test_fscore_50_percent():
    recall = metrics.fscore(tp=10, fp=10, tn=10, fn=10)
    assert recall == 0.5

def test_fscore_zerodivicion():
    assert metrics.fscore(tp=0, fp=10, tn=0, fn=10) == 0

def test_fscore_zero_beta_raises_exeception():
    with pytest.raises(AssertionError):
        _ = metrics.fscore(tp=10, fp=10, tn=10, fn=10, beta=0)