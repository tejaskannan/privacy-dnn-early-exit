import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Tuple

from privddnn.utils.constants import BIG_NUMBER, SMALL_NUMBER
from privddnn.utils.metrics import compute_entropy
from .attack_classifiers import MAJORITY


L1_ERROR = 'l1_error'
L1_ERROR_STD = 'l1_error_std'
L2_ERROR = 'l2_error'
L2_ERROR_STD = 'l2_error_std'
WEIGHTED_L1_ERROR = 'weighted_l1_error'
WEIGHTED_L2_ERROR = 'weighted_l2_error'


class InputGenerator:

    def name(self) -> str:
        raise NotImplementedError()

    def fit(self, exit_decisions: np.ndarray, data_inputs: np.ndarray, data_labels: np.ndarray):
        raise NotImplementedError()

    def predict(self, exit_decisions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def score(self, exit_decisions: np.ndarray, data_inputs: np.ndarray) -> Dict[str, float]:
        """
        Evalutates the given input generation model.

        Args:
            exit_decisions: A [B, T, D] array of exit decisions (D) for each step (T) and input sample (B)
            data_inputs: A [B, ...] array of (average) data inputs for each step
        Returns:
            A dictionary of score metric -> metric value
        """
        assert len(exit_decisions) == len(data_inputs), 'Decisions and Data Inputs are misaligned'
        assert len(exit_decisions.shape) == 3, 'Must provide a 3d array of exit decisions'
        assert len(data_inputs) >= 2, 'Must provide at least a 2d array of data inputs'

        l1_errors: List[float] = []
        l2_errors: List[float] = []
        confidence_scores_list: List[float] = []

        for exit_decision, data_input in zip(exit_decisions, data_inputs):
            predicted_data, confidence_score = self.predict(exit_decision)

            l1_error = np.sum(np.abs(data_input - predicted_data))
            l2_error = np.sum(np.square(data_input - predicted_data))
            confidence_scores_list.append(confidence_score)

            #if l1_error > 200:
            #    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

            #    ax1.imshow(predicted_data)
            #    ax2.imshow(data_input)

            #    ax1.set_title('Prediction')
            #    ax2.set_title('True')

            #    plt.show()

            l1_errors.append(l1_error)
            l2_errors.append(l2_error)

        confidence_scores = np.vstack(confidence_scores_list).reshape(-1)
        normalized_confidence_scores = confidence_scores / np.sum(confidence_scores)

        avg_l1_error = np.average(l1_errors)
        std_l1_error = np.std(l1_errors)
        avg_l2_error = np.average(l2_errors)
        std_l2_error = np.std(l2_errors)

        weighted_l1_error = np.sum(np.multiply(normalized_confidence_scores, l1_errors))
        weighted_l2_error = np.sum(np.multiply(normalized_confidence_scores, l2_errors))

        return {
            L1_ERROR: avg_l1_error,
            L1_ERROR_STD: std_l1_error,
            L2_ERROR: avg_l2_error,
            L2_ERROR_STD: std_l2_error,
            WEIGHTED_L1_ERROR: weighted_l1_error,
            WEIGHTED_L2_ERROR: weighted_l2_error
        }


class MajorityGenerator(InputGenerator):

    def __init__(self):
        self._clf: Dict[Tuple[int, ...], np.ndarray] = dict()
        self._most_freq = np.empty(0)

    @property
    def name(self) -> str:
        return MAJORITY

    def fit(self, exit_decisions: np.ndarray, data_inputs: np.ndarray, data_labels: np.ndarray):
        """
        Fits the exit decisions to the data inputs using the majority occurance
        of exit counts in the training set.

        Args:
            exit_decisions: A [B, T, D] array of exit decisions (D) for each input sample (B) and window step (T)
            data_inputs: A [B, ...] array of data inputs for each sample (B)
            data_labels: A [B] array of labels for each data sample (B)
        """
        assert len(exit_decisions.shape) == 3, 'Must provide a 3d array of exit decisions'
        assert len(data_inputs.shape) >= 2, 'Must provide at least a 2d array of data inputs'
        assert exit_decisions.shape[0] == data_inputs.shape[0], 'Exit Decisions and Data Inputs are misaligned'

        exit_counter: Counter = Counter()
        features_array = np.sum(exit_decisions, axis=1).astype(int)  # [B, D]
        exit_labels: Dict[Tuple[int, ...], List[int]] = dict()
        data_shape = data_inputs.shape[1:]

        for exit_features, data_features, data_label in zip(features_array, data_inputs, data_labels):
            exit_counts = tuple(exit_features)  # D

            if exit_counts not in self._clf:
                self._clf[exit_counts] = np.zeros(shape=data_shape)
                exit_labels[exit_counts] = []

            self._clf[exit_counts] += data_features
            exit_labels[exit_counts].append(data_label)
            exit_counter[exit_counts] += 1

        # Create the average input
        for exit_counts in self._clf.keys():
            self._clf[exit_counts] /= exit_counter[exit_counts]

        # Get the top-occuring count as a base case
        most_common_exit = exit_counter.most_common(1)[0][0]
        self._most_freq = self._clf[most_common_exit]

        self._confidence: Dict[Tuple[int, ...], float] = dict()

        num_labels = np.amax(data_labels) + 1
        max_dist = np.ones(shape=(num_labels, )).astype(float) / num_labels
        max_entropy = compute_entropy(max_dist, axis=-1)

        for exit_counts, labels in exit_labels.items():
            label_counts = np.bincount(labels, minlength=num_labels).astype(float)
            label_dist = label_counts / np.sum(label_counts)
            self._confidence[exit_counts] = 1.0 - max(compute_entropy(label_dist, axis=-1), 0.0) / max_entropy

        #min_count = min(exit_counter.values())

        #for exit_counts in exit_counter.keys():
        #    score = min_count / exit_counter[exit_counts]
        #    self._confidence[exit_counts] = score

        #print(self._confidence)

    def predict(self, exit_decisions: np.ndarray) -> Tuple[np.ndarray, float]:
        assert len(exit_decisions.shape) == 2, 'Must provide a 2d array of exit decisions'
        target = tuple(np.sum(exit_decisions, axis=0))
        best_key = None
        best_diff = BIG_NUMBER

        if target in self._clf:
            prediction = self._clf[target]
            confidence = self._confidence[target]
        else:
            for key in self._clf.keys():
                diff = sum((abs(k - t) for k, t in zip(key, target)))
                if diff < best_diff:
                    best_diff = diff
                    best_key = key

            if best_key is None:
                prediction = self._most_freq
                confidence = 0.0
            else:
                prediction = self._clf.get(best_key, self._most_freq)
                confidence = self._confidence.get(best_key, 0.0)

        return prediction, confidence

