from typing import *

import torch
from sklearn import metrics


def pixelwise_accuracy(prediction: torch.Tensor, target: torch.Tensor, ignore_index: Optional[int] = None):
    predicted_labels, target = _prepare_for_pixelwise_comparison(prediction, target, ignore_index)
    n_equal = (predicted_labels == target).sum()
    accuracy = n_equal.float() / len(target)
    return accuracy.item()


def pixelwise_f1score(prediction: torch.Tensor, target: torch.Tensor, ignore_index: Optional[int] = None):
    predicted_labels, target = _prepare_for_pixelwise_comparison(prediction, target, ignore_index)
    average = 'macro' if len(target.unique()) > 2 else 'binary'
    score = metrics.f1_score(predicted_labels.cpu(), target.cpu(), average=average)
    return score


def pixelwise_recall(prediction: torch.Tensor, target: torch.Tensor, ignore_index: Optional[int] = None):
    predicted_labels, target = _prepare_for_pixelwise_comparison(prediction, target, ignore_index)
    average = 'macro' if len(target.unique()) > 2 else 'binary'
    score = metrics.recall_score(predicted_labels.cpu(), target.cpu(), average=average)
    return score


def pixelwise_precision(prediction: torch.Tensor, target: torch.Tensor, ignore_index: Optional[int] = None):
    predicted_labels, target = _prepare_for_pixelwise_comparison(prediction, target, ignore_index)
    average = 'macro' if len(target.unique()) > 2 else 'binary'
    score = metrics.precision_score(predicted_labels.cpu(), target.cpu(), average=average)
    return score


def _prepare_for_pixelwise_comparison(prediction, target, ignore_index):
    assert prediction.ndimension() == 4, 'assuming 4-dimensional input'
    assert target.ndimension() == 3, 'assuming 3-dimensional target'
    assert prediction.shape[-2:] == target.shape[-2:], 'prediction and target must have same shape'
    predicted_labels = prediction.argmax(dim=1).view(-1)
    target = target.view(-1)
    if ignore_index is not None:
        valid_indexes = target != ignore_index
        target = target[valid_indexes]
        predicted_labels = predicted_labels[valid_indexes]
    return predicted_labels, target