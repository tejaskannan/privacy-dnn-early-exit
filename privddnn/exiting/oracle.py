import numpy as np
from typing import List
from .early_exit import EarlyExiter, EarlyExitResult


class OraclePolicy(EarlyExiter):

    def __init__(self, rates: List[float]):
        assert len(rates) == 2, 'Oracle only supports 2-level models'
        super().__init__(rates=rates)
        self._policy_preds = np.empty(0)
        self._policy_levels = np.empty(0)

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        assert len(val_probs.shape) == 3, 'Must provide a 3d array of probabilities'
        assert len(val_labels.shape) == 1, 'Must provide a 1d array of labels'
        assert val_probs.shape[1] == len(self.rates), 'Misaligned probabilities and rates'
        assert val_probs.shape[0] == val_labels.shape[0], 'Misaligned probabilities and labels'

        preds = np.argmax(val_probs, axis=-1)  # [B, L]
        labels = val_labels  # [B]

        num_labels = np.amax(labels) + 1
        num_samples = labels.shape[0]

        self._policy_preds = np.empty(labels.shape).astype(int)
        self._policy_levels = np.empty(labels.shape).astype(int)
        already_used = np.zeros_like(labels).astype(int)
        stay_counts = np.zeros(shape=(num_labels, ))
        elev_counts = np.zeros(shape=(num_labels, ))

        for pred in range(num_labels):

            for idx in range(num_samples):
                if (stay_rate < self.rates[0]) and (preds[idx, 0] == labels[idx]):
                    self._policy_preds[idx] = pred
                    self._policy_levels[idx] = 0
                    already_used[idx] = 1
                    stay_counts[pred] += 1
                elif (stay_rate > self.rates[0]) and (preds[idx, 1] == labels[idx]):
                    self._policy_preds[idx] = pred
                    self._policy_levels[idx] = 1
                    already_used[idx] = 1
                    elev_counts[pred] += 1

        for idx in range(num_samples):
            if already_used[idx] == 0:
                stay_rates = stay_counts / np.maximum(stay_counts + elev_counts, 1)
                diff = np.abs(stay_rates - self.rates[0])
                max_idx = np.argmax(diff)

                if stay_rates[max_idx] < self.rates[0]:
                    self._policy_preds[idx] = max_idx
                    self._policy_levels[idx] = 0
                    already_used[idx] = 1
                    stay_counts[pred] += 1
                else:
                    self._policy_preds[idx] = max_idx
                    self._policy_levels[idx] = 1
                    already_used[idx] = 1
                    elev_counts[pred] += 1

        is_correct = (self._policy_preds == labels).astype(float)
        print('Accuracy: {}'.format(np.average(is_correct)))
    
        assert np.all(np.isclose(already_used, 1)), 'Must use all samples after fitting'

    def test(self, num_samples: int) -> EarlyExitResult:
        output_counts = np.bincount(self._policy_levels, minlength=self.num_outputs)
        observed_rates = output_counts / num_samples

        return EarlyExitResult(predictions=self._policy_preds,
                               output_levels=self._policy_levels,
                               observed_rates=observed_rates)
