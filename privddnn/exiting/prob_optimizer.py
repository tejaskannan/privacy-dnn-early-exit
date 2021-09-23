import numpy as np
from typing import Tuple

from privddnn.utils.metrics import create_confusion_matrix
from privddnn.utils.constants import SMALL_NUMBER, BIG_NUMBER
from .metric_distribution import MetricDistributions


STEP_SIZE = 1e-3
MAX_ITER = 75
PATIENCE = 15


class LikelihoodLoss:

    def __init__(self, metrics: np.ndarray, preds: np.ndarray, labels: np.ndarray, target: float):
        self._metrics = metrics
        self._preds = preds
        self._labels = labels

        self._target = target
        self._num_labels = np.max(labels) + 1  # L

        # Make the metric distribution
        self._metric_dist = MetricDistributions(preds=preds, metrics=metrics, labels=labels)

        # Get the confusion matrix
        confusion_mat = create_confusion_matrix(predictions=preds, labels=labels)

        # Get the conditional distribution P(pred | label), [L, L]
        self._pred_given_label = confusion_mat / (np.sum(confusion_mat, axis=0, keepdims=True) + SMALL_NUMBER)

        # Get the probability of selecting each threhold given the prediction
        self._threshold_given_pred = confusion_mat / (np.sum(confusion_mat, axis=-1, keepdims=True) + SMALL_NUMBER)

    @property
    def target(self) -> float:
        return self._target

    @property
    def num_labels(self) -> int:
        return self._num_labels

    def __call__(self, thresholds: np.ndarray) -> Tuple[float, np.ndarray]:
        probs = self.compute_probs(thresholds=thresholds)

        abs_diffs = float(np.sum(np.abs(probs - self.target)))
        data_range = float(np.max(probs) - np.min(probs))

        return abs_diffs + data_range, probs

    def compute_probs(self, thresholds: np.ndarray) -> np.ndarray:
        assert len(thresholds.shape) == 1, 'Must provide a 1d thresholds array'

        prob_elev_for_label = np.zeros(shape=(self.num_labels, ))

        for label in range(self.num_labels):
            prob_sum = 0.0

            # Marginalize over the prediction
            for pred in range(self.num_labels):
                p_pred = self._pred_given_label[pred, label]
                p_metric = self._metric_dist.cdf(pred=pred, label=label, value=thresholds[pred])
                prob_sum += p_pred * p_metric

                # Marginalize over the selected threshold
                #for selected_idx in range(self.num_labels):
                #    p_threshold = self._threshold_given_pred[pred, selected_idx]
                #    p_metric = self._metric_dist.cdf(pred=pred, value=thresholds[selected_idx])

                #    prob_sum += p_pred * p_threshold * p_metric

            prob_elev_for_label[label] = prob_sum

        return prob_elev_for_label

    def derivative(self, thresholds: np.ndarray, probs: np.ndarray) -> np.ndarray:
        signs = np.sign(probs - self.target)
        print('Sign: {}'.format(signs))

        gradient = np.zeros_like(thresholds)

        for threshold_idx in range(self.num_labels):
            for label in range(self.num_labels):
                for pred in range(self.num_labels):
                    threshold_pdf = self._metric_dist.approx_pdf(pred=pred, value=thresholds[threshold_idx])
                    p_threshold = self._threshold_given_pred[pred, threshold_idx]
                    p_pred = self._pred_given_label[pred, label]
                    gradient[threshold_idx] += signs[label] * p_threshold * p_pred * threshold_pdf

        print(gradient)

        return gradient


def fit_prob_thresholds(metrics: np.ndarray, preds: np.ndarray, labels: np.ndarray, target: float, start_thresholds: np.ndarray) -> np.ndarray:
    objective = LikelihoodLoss(metrics=metrics,
                               preds=preds,
                               labels=labels,
                               target=target)

    thresholds = np.copy(start_thresholds)
    rand = np.random.RandomState(seed=981029)
    prev_idx = -1
    num_not_improved = 0
    best_loss, prev_probs = objective(thresholds)

    print('Probs: {}, Loss: {}'.format(prev_probs, best_loss))

    for _ in range(MAX_ITER):

        if num_not_improved == 0:
            threshold_idx = np.argmax(np.abs(prev_probs - target))
        else:
            threshold_idx = rand.randint(low=0, high=len(thresholds))

        upper = 1.0
        lower = 0.0
        t = thresholds[threshold_idx]
        best_t = t

        while (upper - lower) > 1e-3:
            thresholds[threshold_idx] = t
            loss, probs = objective(thresholds)

            if loss < best_loss:
                best_t = t
                best_loss = loss

            if probs[threshold_idx] <= target:
                lower = t
            else:
                upper = t

            t = (upper + lower) / 2

        thresholds[threshold_idx] = best_t
        loss, probs = objective(thresholds)
        prev_idx = threshold_idx
        prev_probs = probs

        if abs(best_loss - loss) < SMALL_NUMBER:
            num_not_improved += 1
        else:
            num_not_improved = 0

        #dt = objective.derivative(thresholds=thresholds, probs=probs)
        #dt = dt / (np.linalg.norm(dt, ord=2) + SMALL_NUMBER)

        #thresholds -= STEP_SIZE * dt

        print('Loss: {:.6f}'.format(loss), end='\n')
        print('Probs: {}'.format(probs))

        if num_not_improved >= PATIENCE:
            break

    return loss, thresholds, probs
