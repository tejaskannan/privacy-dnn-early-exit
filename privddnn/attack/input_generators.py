import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Tuple

from privddnn.utils.constants import BIG_NUMBER, SMALL_NUMBER
from privddnn.utils.metrics import compute_entropy
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from .attack_classifiers import MAJORITY


MAE = 'mae'
MAE_STD = 'mae_std'
RMSE = 'rmse'
RMSE_STD = 'rmse_std'
AVG_SSIM = 'avg_ssim'
STD_SSIM = 'std_ssim'
AVG_PSNR = 'avg_psnr'
WEIGHTED_MAE = 'weighted_mae'
WEIGHTED_RMSE = 'weighted_rmse'


class InputGenerator:

    def name(self) -> str:
        raise NotImplementedError()

    def fit(self, exit_decisions: np.ndarray, data_inputs: np.ndarray, data_labels: np.ndarray):
        raise NotImplementedError()

    def predict(self, exit_decisions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def score(self, exit_decisions: np.ndarray, data_inputs: np.ndarray, data_labels: np.ndarray) -> Dict[str, float]:
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

        mae_list: List[float] = []
        rmse_list: List[float] = []
        ssim_scores: List[float] = []
        psnr_scores: List[float] = []
        confidence_scores_list: List[float] = []

        label_error_counter: Counter = Counter()
        label_counter: Counter = Counter()

        for exit_decision, data_input, data_label in zip(exit_decisions, data_inputs, data_labels):
            predicted_data, confidence_score = self.predict(exit_decision)

            mae = np.average(np.abs(data_input - predicted_data))
            rmse = np.sqrt(np.average(np.square(data_input - predicted_data)))
            #psnr_score = psnr(predicted_data, data_input, data_range=255)

            if data_input.shape[-1] == 1:
                data_input = np.squeeze(data_input, axis=-1)
                predicted_data = np.squeeze(predicted_data, axis=-1)
                ssim_score = ssim(predicted_data, data_input, data_range=255)
            else:
                ssim_score = ssim(predicted_data, data_input, data_range=255, multichannel=True)

            label_error_counter[data_label] += ssim_score
            label_counter[data_label] += 1
            
            mae_list.append(mae)
            rmse_list.append(rmse)
            confidence_scores_list.append(confidence_score)
            ssim_scores.append(ssim_score)
            #psnr_scores.append(psnr_score)

            #if confidence_score > 0.7:
            #    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

            #    ax1.imshow(predicted_data)
            #    ax2.imshow(data_input)

            #    ax1.set_title('Prediction')
            #    ax2.set_title('True')

            #    plt.show()

            #l1_errors.append(l1_error)
            #l2_errors.append(l2_error)

        confidence_scores = np.vstack(confidence_scores_list).reshape(-1)
        normalized_confidence_scores = confidence_scores / np.sum(confidence_scores)

        avg_mae = np.average(mae_list)
        std_mae = np.std(mae_list)
        avg_rmse = np.average(rmse_list)
        std_rmse = np.std(rmse_list)
        avg_ssim = np.average(ssim_scores)
        #avg_psnr = np.average(psnr_scores)

        for label in sorted(label_error_counter.keys()):
            avg_error = label_error_counter[label] / label_counter[label]
            print('{} -> {:.4f}'.format(label, avg_error))

        fig, ax = plt.subplots()
        ax.scatter(confidence_scores, mae_list)
        plt.show()

        print('Avg SSIM: {:.4f}'.format(avg_ssim))
        #print('Avg PSNR: {:.4f}'.format(avg_psnr))
        print('Avg MAE: {:.4f}'.format(avg_mae))

        weighted_mae = np.sum(np.multiply(normalized_confidence_scores, mae_list))
        weighted_rmse = np.sum(np.multiply(normalized_confidence_scores, rmse_list))

        return {
            MAE: avg_mae,
            RMSE: avg_rmse,
            MAE_STD: std_mae,
            RMSE_STD: std_rmse,
            AVG_SSIM: avg_ssim,
            WEIGHTED_MAE: weighted_mae,
            WEIGHTED_RMSE: weighted_rmse
        }


class MajorityGenerator(InputGenerator):

    def __init__(self):
        self._clf: Dict[Tuple[int, ...], np.ndarray] = dict()
        self._most_freq = np.empty(0)
        self._rand = np.random.RandomState(seed=5489)

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
                self._clf[exit_counts] = []
                exit_labels[exit_counts] = []

            self._clf[exit_counts].append(data_features)
            exit_labels[exit_counts].append(data_label)
            exit_counter[exit_counts] += 1

        # Create the average input
        #for exit_counts in self._clf.keys():
        #    self._clf[exit_counts] /= exit_counter[exit_counts]

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

        print(self._confidence)

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
            candidates = self._clf[target]
            confidence = self._confidence[target]
        else:
            for key in self._clf.keys():
                diff = sum((abs(k - t) for k, t in zip(key, target)))
                if diff < best_diff:
                    best_diff = diff
                    best_key = key

            if best_key is None:
                candidates = self._most_freq
                confidence = 0.0
            else:
                candidates = self._clf.get(best_key, self._most_freq)
                confidence = self._confidence.get(best_key, 0.0)

        prediction_idx = self._rand.randint(low=0, high=len(candidates))
        prediction = candidates[prediction_idx]

        return prediction, confidence

