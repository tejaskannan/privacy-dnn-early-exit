import numpy as np
import scipy.stats as stats
from collections import defaultdict
from typing import DefaultDict, List


DELTA = 1e-4


class MetricDistributions:

    def __init__(self, metrics: np.ndarray, preds: np.ndarray, labels: np.ndarray):
        assert metrics.shape == preds.shape, 'Misaligned metrics and predictions'

        self._metric_dict: DefaultDict[int, List[float]] = defaultdict(list)  # Pred -> List of metric values

        for metric, pred, label in zip(metrics, preds, labels):
            key = (pred, label)
            self._metric_dict[key].append(metric)

    def cdf(self, pred: int, label: int, value: float) -> float:
        key = (pred, label)
        percentile = stats.percentileofscore(a=self._metric_dict[key],
                                             score=value,
                                             kind='weak')
        return percentile / 100.0

    def approx_pdf(self, pred: int, value: float) -> float:
        lower = self.cdf(pred=pred, value=(value - DELTA))
        upper = self.cdf(pred=pred, value=(value + DELTA))

        return (upper - lower) / (2.0 * DELTA)
