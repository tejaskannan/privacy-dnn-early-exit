import numpy as np
import scipy.stats as stats

from privddnn.utils.metrics import create_confusion_matrix, create_metric_distributions


class LikelihoodLoss:

    def __init__(self,
                 metric_means: np.ndarray,
                 metric_stds: np.ndarray,
                 pred_label_dist: np.ndarray,
                 pred_fractions: np.ndarray,
                 target: float,
                 num_labels: int):
        self._metric_means = metric_means  # [L, L]
        self._metric_stds = metric_stds  # [L, L]
        self._pred_label_dist = pred_label_dist  # [L, L]
        self._pred_fractions = pred_fractions  # [L]
        self._target = target
        self._num_labels = num_labels

    def target(self) -> float:
        return self._target

    def num_labels(self) -> int:
        return self._num_labels

    def __call__(self, thresholds: np.ndarray) -> float:
        probs = self.compute_probs(thresholds=thresholds)
        return float(np.sum(np.abs(probs - self.target)))

    def compute_probs(self, thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert len(thresholds.shape) == 1, 'Must provide a 1d thresholds array'

        prob_elev_for_label = np.zeros(shape=(self.num_labels, ))
        threshold_derivative = np.zeros_like(thresholds)

        for label in range(self.num_labels):
            prob_sum = 0.0

            # Marginalize over the prediction
            for pred in range(self.num_labels):

                # Marginalize over the selected threshold
                for selected_idx in range(self.num_labels):
                    metric_prob = stats.norm.cdf(x=thresholds[selected_idx],
                                                 loc=self._metric_means[pred, label],
                                                 scale=self._metric_stds[pred, label])

                    metric_pdf = stats.norm.pdf(x=thresholds[selected_idx],
                                                loc=self._metric_means[pred, label],
                                                scale=self._metric_stds[pred, label])

                    prob = self._pred_label_dist[pred, label] * metric_prob
                    prob_sum += self._pred_fractions[pred] * prob

            prob_elev_for_label[label] = prob_sum

    return prob_elev_for_label


def fit_prob_thresholds(metrics: np.ndarray, preds: np.ndarray, labels: np.ndarray, target: float) -> np.ndarray:
    num_labels = np.max(labels) + 1

    # Get the fraction of each prediction
    pred_counts = np.bincount(preds, minlength=num_labels)
    pred_fractions = pred_counts / np.sum(pred_counts)  # [L]

    # Get the (normalized) confusion matrix
    confusion_mat = create_confusion_matrix(predictions=preds, labels=labels)

    # Get the Gaussian distribution parameters for each metric | (pred, label) distribution
    means, stds = create_metric_distributions(predictions=preds, labels=labels, metrics=metrics)

    loss_fn = LikelihoodLoss(metric_means=means,
                             metric_stds=stds,
                             pred_label_dist=confusion_mat,
                             pred_fractions=pred_fractions,
                             target=target)

