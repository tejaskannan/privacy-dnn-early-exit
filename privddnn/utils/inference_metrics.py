import numpy as np
from enum import Enum, auto
from typing import Union, List

from .metrics import compute_accuracy, compute_mutual_info, compute_entropy
from .ngrams import create_ngrams, create_ngram_counts


NGRAM_CLUSTERS = 32


class InferenceMetric(Enum):
    ACCURACY = auto()
    MUTUAL_INFORMATION = auto()
    AVG_EXIT = auto()
    NGRAM_MI = auto()
    COUNT_NGRAM_MI = auto()


def compute_windowed_mutual_information(preds: np.ndarray, exit_decisions: np.ndarray, metric: InferenceMetric, num_outputs: int, window_size: int) -> float:
    assert metric in (InferenceMetric.NGRAM_MI, InferenceMetric.COUNT_NGRAM_MI), 'Metric must be NGRAM_MI or COUNT_NGRAM_MI'

    mut_info_scores: List[float] = []

    for ngram_size in range(2, window_size + 1):
        if metric == InferenceMetric.NGRAM_MI:
            ngram_decisions, ngram_preds = create_ngrams(levels=exit_decisions,
                                                         preds=preds,
                                                         n=ngram_size,
                                                         num_outputs=num_outputs,
                                                         num_clusters=NGRAM_CLUSTERS)
        elif metric == InferenceMetric.COUNT_NGRAM_MI:
            ngram_decisions, ngram_preds = create_ngram_counts(levels=exit_decisions,
                                                               preds=preds,
                                                               n=ngram_size,
                                                               num_outputs=num_outputs,
                                                               num_clusters=NGRAM_CLUSTERS)
        else:
            raise ValueError('Unknown metric {}'.format(metric))

        mutual_information = compute_mutual_info(X=ngram_decisions, Y=ngram_preds, should_normalize=True, should_bias_correct=True)
        mut_info_scores.append(mutual_information)

    return float(np.max(mut_info_scores))


def compute_metric(preds: np.ndarray, exit_decisions: np.ndarray, labels: np.ndarray, metric: InferenceMetric, num_outputs: int, window_size: int) -> float:
    if metric == InferenceMetric.ACCURACY:
        accuracy = compute_accuracy(predictions=preds, labels=labels) * 100.0
        return float(accuracy)
    elif metric == InferenceMetric.MUTUAL_INFORMATION:
        mutual_information = compute_mutual_info(X=exit_decisions, Y=preds, should_normalize=True, should_bias_correct=True)
        return float(mutual_information)
    elif metric in (InferenceMetric.NGRAM_MI, InferenceMetric.COUNT_NGRAM_MI):
        return compute_windowed_mutual_information(preds=preds,
                                                   exit_decisions=exit_decisions,
                                                   metric=metric,
                                                   num_outputs=num_outputs,
                                                   window_size=window_size)
    elif metric == InferenceMetric.AVG_EXIT:
        return float(np.average(exit_decisions))
    else:
        raise ValueError('Unknown metric name: {}'.format(metric))
