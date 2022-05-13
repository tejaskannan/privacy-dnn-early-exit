import numpy as np
from enum import Enum, auto
from typing import Union, List

from .metrics import compute_accuracy, compute_mutual_info, compute_entropy
from .ngrams import create_ngrams, create_ngram_counts


class InferenceMetric(Enum):
    ACCURACY = auto()
    MUTUAL_INFORMATION = auto()
    AVG_EXIT = auto()
    NGRAM_MI = auto()
    COUNT_NGRAM_MI = auto()


def compute_metric(preds: np.ndarray, exit_decisions: np.ndarray, labels: np.ndarray, metric: InferenceMetric, num_outputs: int, window_size: int) -> float:
    if metric == InferenceMetric.ACCURACY:
        accuracy = compute_accuracy(predictions=preds, labels=labels) * 100.0
        return float(accuracy)
    elif metric == InferenceMetric.MUTUAL_INFORMATION:
        mutual_information = compute_mutual_info(X=exit_decisions, Y=preds)
        return float(mutual_information)
    elif metric == InferenceMetric.NGRAM_MI:
        ngram_decisions, ngram_preds = create_ngrams(levels=exit_decisions,
                                                     preds=preds,
                                                     n=window_size,
                                                     num_outputs=num_outputs)
        ngram_mi = compute_mutual_info(X=ngram_decisions, Y=ngram_preds)
        return float(ngram_mi)
    elif metric == InferenceMetric.COUNT_NGRAM_MI:
        ngram_decisions, ngram_preds = create_ngram_counts(levels=exit_decisions,
                                                           preds=preds,
                                                           n=window_size,
                                                           num_outputs=num_outputs)
        ngram_mi = compute_mutual_info(X=ngram_decisions, Y=ngram_preds)
        return float(ngram_mi)
    elif metric == InferenceMetric.AVG_EXIT:
        return float(np.average(exit_decisions))
    else:
        raise ValueError('Unknown metric name: {}'.format(metric))




