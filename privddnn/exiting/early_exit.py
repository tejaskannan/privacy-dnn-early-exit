import numpy as np
import os.path
import time
from collections import namedtuple, defaultdict, Counter, deque
from enum import Enum, auto
from sklearn.ensemble import AdaBoostClassifier
from typing import List, Tuple, Dict, DefaultDict, Optional

from privddnn.dataset.data_iterators import DataIterator
from privddnn.utils.metrics import compute_entropy, create_confusion_matrix, sigmoid
from privddnn.utils.metrics import compute_max_prob_metric, compute_entropy_metric, linear_step
from privddnn.utils.np_utils import mask_non_max
from privddnn.utils.constants import BIG_NUMBER, SMALL_NUMBER
from privddnn.utils.file_utils import read_json
from privddnn.controllers.runtime_controller import RandomnessController
from .even_optimizer import fit_thresholds, fit_threshold_randomization
from .prob_optimizer import fit_prob_thresholds
from .threshold_optimizer import fit_thresholds_grad, fit_randomization, fit_sharpness
from .random_optimizer import fit_even_randomization


EarlyExitResult = namedtuple('EarlyExitResult', ['predictions', 'output_levels', 'labels', 'observed_rates', 'num_changed', 'selection_counts'])


class ExitStrategy(Enum):
    RANDOM = auto()
    FIXED = auto()
    ENTROPY = auto()
    MAX_PROB = auto()
    LABEL_ENTROPY = auto()
    LABEL_MAX_PROB = auto()
    HYBRID_MAX_PROB = auto()
    HYBRID_ENTROPY = auto()
    GREEDY_EVEN = auto()
    EVEN_MAX_PROB = auto()
    EVEN_LABEL_MAX_PROB = auto()
    BUFFERED_MAX_PROB = auto()
    DELAYED_MAX_PROB = auto()
    ADAPTIVE_RANDOM_MAX_PROB = auto()
    ROLLING_MAX_PROB = auto()


class SelectionType(Enum):
    GREEDY = auto()
    POLICY = auto()
    RANDOM = auto()


class EarlyExiter:

    def __init__(self, rates: List[float]):
        assert np.isclose(np.sum(rates), 1.0), 'Rates must sum to 1'
        self._rates = rates

    @property
    def rates(self) -> List[float]:
        return self._rates

    @property
    def num_outputs(self) -> int:
        return len(self._rates)

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        pass

    def select_output(self, probs: np.ndarray) -> Tuple[int, SelectionType]:
        raise NotImplementedError()

    def update(self, first_pred: int, pred: int, level: int):
        pass

    def reset(self, num_labels: int, pred_rates: np.ndarray):
        pass

    def get_prediction(self, probs: np.ndarray, level: int) -> Tuple[int, bool]:
        return np.argmax(probs[level]), False

    def test(self, data_iterator: DataIterator,
                   num_labels: int,
                   pred_rates: np.ndarray,
                   max_num_samples: Optional[int]) -> EarlyExitResult:
        predictions: List[int] = []
        output_levels: List[int] = []
        labels: List[int] = []

        self.reset(num_labels=num_labels, pred_rates=pred_rates)

        num_changed = 0
        num_samples = 0
        selection_counts: DefaultDict[SelectionType, int] = defaultdict(int)

        for _, sample_probs, label in data_iterator:
            if (max_num_samples is not None) and (num_samples >= max_num_samples):
                break

            level, selection_type = self.select_output(probs=sample_probs)
            pred, did_change = self.get_prediction(probs=sample_probs, level=level)

            first_pred = np.argmax(sample_probs[0], axis=-1)
            self.update(first_pred=first_pred, pred=pred, level=level)

            selection_counts[selection_type] += 1
            num_changed += int(did_change)

            predictions.append(pred)
            output_levels.append(level)
            labels.append(label)
            num_samples += 1

            #print('Level: {}, Label: {}'.format(level, label))

        output_counts = np.bincount(output_levels, minlength=self.num_outputs)
        observed_rates = output_counts / num_samples

        return EarlyExitResult(predictions=np.vstack(predictions).reshape(-1),
                               output_levels=np.vstack(output_levels).reshape(-1),
                               labels=np.vstack(labels).reshape(-1),
                               observed_rates=observed_rates,
                               selection_counts=selection_counts,
                               num_changed=num_changed)


class RandomExit(EarlyExiter):

    def __init__(self, rates: List[float]):
        super().__init__(rates=rates)
        self._output_idx = list(range(len(rates)))

    def select_output(self, probs: np.ndarray) -> Tuple[int, SelectionType]:
        level = np.random.choice(self._output_idx, size=1, p=self.rates)
        return int(level), SelectionType.POLICY


class GreedyEvenExit(EarlyExiter):

    def __init__(self, rates: List[float]):
        super().__init__(rates=rates)
        self._stay_counter: Counter = Counter()
        self._elevate_counter: Counter = Counter()

    def update(self, first_pred: int, pred: int, level: int):
        if level == 0:
            self._stay_counter[pred] += 1
        else:
            self._elevate_counter[pred] += 1

    def reset(self, num_labels: int, pred_rates: np.ndarray):
        self._stay_counter = Counter()
        self._elevate_counter = Counter()

    def select_output(self, probs: np.ndarray) -> Tuple[int, SelectionType]:
        first_pred = np.argmax(probs[0])
        stay_rate = self._stay_counter[first_pred] / (self._stay_counter[first_pred] + self._elevate_counter[first_pred] + SMALL_NUMBER)
        greedy_level = int(stay_rate > self.rates[0])

        if np.isclose(self.rates[0], 0.0):
            return 1, SelectionType.POLICY
        elif np.isclose(self.rates[0], 1.0):
            return 0, SelectionType.POLICY
        else:
            return greedy_level, SelectionType.POLICY


class DelayedExiter(EarlyExiter):

    def __init__(self, rates: List[float], window_size: int, delay_prob: float):
        super().__init__(rates=rates)
        assert delay_prob > 0.0 and delay_prob < 1.0, 'The delay probability must be in (0, 1).'

        self._thresholds: List[List[float]] = []
        self._threshold_rates = np.arange(0.0, 1.01, 0.05)
        self._window_size = window_size
        self._delay_prob = delay_prob
        self._rand = np.random.RandomState(seed=3958)

        self._exit_counter: Counter = Counter()
        self._elev_counter: Counter = Counter()
        self._pred_counts = np.empty((1, 1))

    @property
    def num_threshold_rates(self) -> int:
        return len(self._thresholds)

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def delay_prob(self) -> float:
        return self._delay_prob

    def get_thresholds_for_rate(self, rate: float) -> List[float]:
        # Get the nearest rates on either side of the target
        nearest_rate_idx = np.argmin(np.abs(self._threshold_rates - rate))

        # Get the nearest indices on each side
        if self._threshold_rates[nearest_rate_idx] < rate:
            lower_idx = nearest_rate_idx
            upper_idx = (nearest_rate_idx + 1) if nearest_rate_idx < (self.num_threshold_rates - 1) else nearest_rate_idx
        else:
            lower_idx = (nearest_rate_idx - 1) if nearest_rate_idx > 0 else 0
            upper_idx = nearest_rate_idx

        # Randomly select which thresholds to use
        r = self._rand.uniform()
        if r < 0.5:
            return self._thresholds[lower_idx]

        return self._thresholds[upper_idx]

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        super().fit(val_probs=val_probs, val_labels=val_labels)

        assert val_probs.shape[0] == val_labels.shape[0], 'misaligned probabilites and labels'
        assert self.num_outputs == 2, 'threshold exiting only works with 2 outputs'
        assert val_probs.shape[1] == self.num_outputs, 'expected {} outputs. got {}'.format(self.num_outputs, val_probs.shape[1])

        # Compute the maximum probability for each predicted distribution
        metrics = self.compute_metric(probs=val_probs)  # [B, L]

        # Compute the thresholds based on the quantile for each potential rate
        for threshold_rate in self._threshold_rates:
            if np.isclose(threshold_rate, 1.0):
                t = 0.0
            elif np.isclose(threshold_rate, 0.0):
                t = 1.0
            else:
                t = np.quantile(metrics[:, 0], q=1.0 - threshold_rate)

            thresholds = [t, 0.0]
            self._thresholds.append(thresholds)

    def reset(self, num_labels: int, pred_rates: np.ndarray):
        self._exit_counter = Counter()
        self._elev_counter = Counter()
        self._pred_counts = np.eye(num_labels)

    def select_output(self, probs: np.ndarray, remaining_exit: int, remaining_window: int) -> Tuple[int, SelectionType]:
        first_pred = np.argmax(probs[0])

        exit_rate = float(remaining_exit) / float(remaining_window)

        # Compute the expected exit count for this prediction
        total_count = self._exit_counter[first_pred] + self._elev_counter[first_pred] + 1
        expected_exit = self.rates[0] * total_count
        exit_cost = (self._exit_counter[first_pred] + 1) - expected_exit

        # Compute the expected elevation count based on the confusion matrix
        pred_rates = self._pred_counts[first_pred] / np.sum(self._pred_counts[first_pred])
        expected_elev = sum((pred_rates[pred] * self.rates[1] * (self._exit_counter[pred] + self._elev_counter[pred]) for pred in range(len(pred_rates))))
        observed_elev = sum((pred_rates[pred] * self._elev_counter[pred] for pred in range(len(pred_rates))))
        elev_cost = observed_elev - expected_elev

        # Perform data-dependent selection
        thresholds = self.get_thresholds_for_rate(rate=exit_rate)
        metrics = self.compute_metric(probs)
        policy_level = int(metrics[0] < thresholds[0])

        if np.isclose(exit_rate, 0.0):
            return 1, SelectionType.GREEDY
        elif np.isclose(exit_rate, 1.0):
            return 0, SelectionType.GREEDY
        elif (exit_cost >= self.window_size) and (elev_cost >= self.window_size):
            return 0 if exit_cost < elev_cost else 1, SelectionType.GREEDY
        elif (exit_cost >= self.window_size):
            return 1, SelectionType.GREEDY
        elif (elev_cost >= self.window_size):
            return 0, SelectionType.GREEDY
        else:
            return policy_level, SelectionType.POLICY

    def update(self, first_pred: int, pred: int, level: int):
        if level == 0:
            self._exit_counter[pred] += 1
        else:
            self._elev_counter[pred] += 1
            self._pred_counts[first_pred, pred] += 1

    def test(self, data_iterator: DataIterator,
                   num_labels: int,
                   pred_rates: np.ndarray,
                   max_num_samples: Optional[int]) -> EarlyExitResult:
        predictions: List[int] = []
        output_levels: List[int] = []
        labels: List[int] = []

        self.reset(num_labels=num_labels, pred_rates=pred_rates)

        num_changed = 0
        num_samples = 0
        selection_counts: DefaultDict[SelectionType, int] = defaultdict(int)

        prev_preds: np.ndarray = np.empty(1)
        prev_level: int = -1
        prev_label: int = -1

        window_count = 0
        window_exit_count = 0
        observed_exit_count = 0

        for _, sample_probs, label in data_iterator:
            if (max_num_samples is not None) and (num_samples >= max_num_samples):
                break

            if window_count == 0:
                observed_exit_count = 0
                exit_count = self.rates[0] * self.window_size
                window_exit_count = int(exit_count) + int(self._rand.uniform() < (exit_count - int(exit_count)))

            level, selection_type = self.select_output(probs=sample_probs,
                                                       remaining_exit=(window_exit_count - observed_exit_count),
                                                       remaining_window=(self.window_size - window_count))
            preds = np.argmax(sample_probs, axis=-1)  # [L]

            observed_exit_count += int(level == 0)
            window_count += 1

            if num_samples == 0:
                prev_preds = preds
                prev_level = level
                prev_label = label

                num_samples += 1
                selection_counts[selection_type] += 1
                continue

            # Perform delayed exiting
            r = self._rand.uniform()

            if r < self.delay_prob:
                current_first_pred = prev_preds[0]
                current_pred = prev_preds[prev_level]
                current_level = prev_level
                current_label = prev_label

                prev_preds = preds
                prev_level = level
                prev_label = label
            else:
                current_first_pred = preds[0]
                current_pred = preds[level]
                current_level = level
                current_label = label

            if window_count == self.window_size:
                window_count = 0

            self.update(first_pred=current_first_pred,
                        pred=current_pred,
                        level=current_level)

            selection_counts[selection_type] += 1
            #num_changed += int(did_change)

            predictions.append(current_pred)
            output_levels.append(current_level)
            labels.append(current_label)
            num_samples += 1

        # Include the final sample
        predictions.append(prev_preds[prev_level])
        output_levels.append(prev_level)
        labels.append(prev_label)

        # Compute aggregate stats
        output_counts = np.bincount(output_levels, minlength=self.num_outputs)
        observed_rates = output_counts / num_samples

        #print('Levels: {}'.format(output_levels))
        #print('Predictions: {}'.format(predictions))
        #print('Observed Elev Rate: {}'.format(np.average(output_levels)))

        return EarlyExitResult(predictions=np.vstack(predictions).reshape(-1),
                               output_levels=np.vstack(output_levels).reshape(-1),
                               labels=np.vstack(labels).reshape(-1),
                               observed_rates=observed_rates,
                               selection_counts=selection_counts,
                               num_changed=num_changed)


class DelayedMaxProb(DelayedExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_max_prob_metric(probs=probs)


class ThresholdExiter(EarlyExiter):

    def __init__(self, rates: List[float]):
        super().__init__(rates=rates)
        self._thresholds = [0.0 for _ in rates]

    @property
    def thresholds(self) -> List[float]:
        return self._thresholds

    def compute_metric(self, probs: np.ndarray) -> float:
        raise NotImplementedError()

    def get_quantile(self, level: int) -> float:
        return 1.0 - self.rates[level]

    def set_threshold(self, t: float, level: int):
        assert level >= 0 and level < self.num_outputs, 'Level must be in [0, {})'.format(self.num_outputs)
        self._thresholds[level] = t

    def select_output(self, probs: np.ndarray) -> Tuple[int, SelectionType]:
        metric = self.compute_metric(probs)
        comp = np.greater(metric, self.thresholds).astype(int)  # [L]
        return np.argmax(comp), SelectionType.POLICY  # Breaks ties by selecting the first value

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        super().fit(val_probs=val_probs, val_labels=val_labels)

        assert val_probs.shape[0] == val_labels.shape[0], 'misaligned probabilites and labels'
        assert self.num_outputs == 2, 'threshold exiting only works with 2 outputs'
        assert val_probs.shape[1] == self.num_outputs, 'expected {} outputs. got {}'.format(self.num_outputs, val_probs.shape[1])

        # Compute the maximum probability for each predicted distribution
        metrics = self.compute_metric(probs=val_probs)  # [B, L]

        # Comute the thresholds based on the quantile
        # TODO: Make this 'cumulative' by removing the samples stopped at the previous level(s)
        for level in range(self.num_outputs):
            if np.isclose(self.rates[level], 1.0):
                t = 0.0
            elif np.isclose(self.rates[level], 0.0):
                t = 1.0
            else:
                t = np.quantile(metrics[:, level], q=1.0 - self.rates[level])

            self.set_threshold(t, level)

        # Catch everything at the last level
        self.set_threshold(0.0, self.num_outputs - 1)


class EntropyExit(ThresholdExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_entropy_metric(probs=probs)


class MaxProbExit(ThresholdExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_max_prob_metric(probs=probs)


class BufferedExiter(ThresholdExiter):

    def __init__(self, epsilon: float, rates: List[float], window_size: int, pred_window: int):
        super().__init__(rates=rates)
        assert epsilon < (1.0 / window_size), 'Epsilon must be at most (1 / W).'

        self._window_size = window_size
        self._pred_window = pred_window
        self._epsilon = epsilon

        self._stay_counter: Counter = Counter()
        self._elevate_counter: Counter = Counter()
        self._prior = 10
        self._pred_counts = np.zeros(shape=(1, 1))

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def pred_window(self) -> int:
        return self._pred_window

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def update(self, first_pred: int, pred: int, level: int):
        if level == 0:
            self._stay_counter[pred] += 1
        else:
            self._elevate_counter[pred] += 1
            self._pred_counts[first_pred, pred] += 1

    def get_prediction(self, probs: np.ndarray, level: int) -> Tuple[int, bool]:
        level_probs = probs[level]  # [K]
        pred = np.argmax(level_probs)
        first_pred = np.argmax(probs[0])

        r = np.random.uniform()
        if (r < (1.0 - self._epsilon)):
            return pred, False

        # Make sure the prediction does not go out of the hard bound
        stay_count = self._stay_counter[pred]
        elevate_count = self._elevate_counter[pred]
        total_count = stay_count + elevate_count + 1  # Account for this sample

        expected_stay = self._rates[0] * total_count
        expected_elevate = self._rates[1] * total_count

        if (level == 0) and (abs(expected_stay - (stay_count + 1)) > self.pred_window):
            sorted_preds = np.argsort(level_probs)[::-1]

            for i in range(1, len(sorted_preds)):
                stay_count = self._stay_counter[sorted_preds[i]] + 1
                total_count = stay_count + self._elevate_counter[sorted_preds[i]]
                expected_stay = self._rates[0] * total_count

                if abs(expected_stay - stay_count) <= self.pred_window:
                    return sorted_preds[i], True

        elif (level == 1) and (abs(expected_elevate - (elevate_count + 1)) > self.pred_window):
            elevate_count = self._elevate_counter[first_pred] + 1
            total_count = elevate_count + self._stay_counter[first_pred]
            expected_elevate = self._rates[1] * total_count

            if abs(elevate_count - expected_elevate) <= self.pred_window:
                return first_pred, True

            sorted_preds = np.argsort(level_probs)[::-1]

            for i in range(1, len(sorted_preds)):
                elevate_count = self._elevate_counter[sorted_preds[i]] + 1
                total_count = elevate_count + self._stay_counter[sorted_preds[i]]
                expected_elevate = self._rates[1] * total_count

                if abs(expected_elevate - elevate_count) <= self.pred_window:
                    return sorted_preds[i], True

        return pred, False

    def reset(self, num_labels: int, pred_rates: np.ndarray):
        super().reset(num_labels=num_labels, pred_rates=pred_rates)
        self._pred_counts = np.eye(num_labels) * self._prior

    def test(self, data_iterator: DataIterator,
                   num_labels: int,
                   pred_rates: np.ndarray,
                   max_num_samples: Optional[int]) -> EarlyExitResult:
        #num_samples, num_outputs, num_labels = test_probs.shape
        #assert num_outputs == self.num_outputs, 'Expected {} outputs. Got {}'.format(self.num_outputs, num_outputs)
        #assert pred_rates.shape == (num_outputs, num_labels), 'Exit rates should be a ({}, {}) array. Got {}'.format(num_outputs, num_labels, pred_rates.shape)

        predictions: List[int] = []
        output_levels: List[int] = []
        labels: List[int] = []

        window_data: List[np.ndarray] = []
        window_labels: List[int] = []

        self.reset(num_labels=num_labels, pred_rates=pred_rates)

        elev_count = int(self.window_size * self.rates[1])
        elev_remainder = (self.window_size * self.rates[1]) - elev_count

        num_changed = 0
        num_samples = 0
        selection_counts: DefaultDict[SelectionType, int] = defaultdict(int)

        for _, sample_probs, label in data_iterator:
            if (max_num_samples is not None) and (num_samples >= max_num_samples):
                break

            window_data.append(sample_probs)
            window_labels.append(label)

            if len(window_data) == self.window_size:
                confidence_scores = np.array([s[0] for s in map(self.compute_metric, window_data)])

                window_count = int(min(elev_count + int(np.random.uniform() < elev_remainder), self.window_size))
                window_indices = np.arange(self.window_size)

                result_levels = np.zeros(self.window_size, dtype=int)
                result_preds = np.zeros(self.window_size, dtype=int) - 1
                result_changed = np.zeros(self.window_size)

                if np.isclose(self.rates[1], 1.0):
                    selected_indices = window_indices
                elif np.isclose(self.rates[1], 0.0):
                    selected_indices = []
                else:
                    selected_indices = np.argsort(confidence_scores)[0:window_count].astype(int).tolist()

                    # Get the `distance` to the stay rate for each 1st-level prediction
                    #total_counts = np.array([self._stay_counter[pred] + self._elevate_counter[pred] + self._prior for pred in range(num_labels)])

                    #observed_stay = np.array([self._stay_counter[pred] + self.rates[0] * self._prior for pred in range(num_labels)])
                    #expected_stay = self.rates[0] * total_counts

                    #observed_elev = np.array([self._elevate_counter[pred] + self.rates[1] * self._prior for pred in range(num_labels)])
                    #expected_elev = self.rates[1] * total_counts

                    ## Sort the window indices based on the confidence scores
                    #sorted_indices = np.argsort(confidence_scores)

                    #selected_indices: List[int] = []
                    #selected_mask = np.ones(self.window_size)

                    #pred_rates = self._pred_counts / np.sum(self._pred_counts, axis=-1, keepdims=True)

                    #for window_idx in sorted_indices:
                    #    if len(selected_indices) >= window_count:
                    #        break

                    #    first_pred = np.argmax(window_data[window_idx][0])
                    #    obs_count = np.sum(pred_rates[first_pred] * observed_elev)
                    #    expected_count = np.sum(pred_rates[first_pred] * expected_elev)

                    #    if (observed_stay[first_pred] > (expected_stay[first_pred] - self.window_size)) and (obs_count < (expected_count + self.window_size)):
                    #        selected_indices.append(window_idx)
                    #        selected_mask[window_idx] = 0

                    #if len(selected_indices) < window_count:
                    #    # Compute the difference factors we use to scale the confidence scores
                    #    diff_factors = np.empty(self.window_size)
                    #    for window_idx in range(self.window_size):
                    #        first_pred = np.argmax(window_data[window_idx][0])
                    #        obs_count = np.sum(pred_rates[first_pred] * observed_elev)
                    #        expected_count = np.sum(pred_rates[first_pred] * expected_elev)

                    #        diff_factors[window_idx] = (expected_count / obs_count)

                    #    # Create the sample probabilities based on confidence scores
                    #    scaled_confidence_scores = (1.0 - confidence_scores) * diff_factors
                    #    score_sum = np.sum(scaled_confidence_scores)
                    #    sample_probs = self._epsilon + (1.0 - self.window_size * self._epsilon) * scaled_confidence_scores / score_sum

                    #    sample_probs = np.ones(shape=(self.window_size, )) / self.window_size

                    #    # Re-normalize after masking
                    #    sample_probs *= selected_mask
                    #    sample_probs /= np.sum(sample_probs)

                    #    selected = np.random.choice(window_indices, size=window_count - len(selected_indices), p=sample_probs)
                    #    selected_indices.extend(selected)

                for idx in range(self.window_size):
                    level = int(idx in selected_indices)
                    pred = np.argmax(window_data[idx][level])
                    did_change = False
                    #pred, did_change = self.get_prediction(probs=window_data[idx], level=level)
                    first_pred = np.argmax(window_data[idx][0])

                    self.update(first_pred=first_pred, pred=pred, level=level)

                    result_levels[idx] = level
                    result_preds[idx] = pred
                    result_changed[idx] = int(did_change)

                    num_changed += int(did_change)

                # Make each window order-invariant
                sample_idx = np.arange(self.window_size)
                np.random.shuffle(sample_idx)

                for idx in sample_idx:
                    selection_counts[SelectionType.POLICY] += 1

                    predictions.append(result_preds[idx])
                    labels.append(window_labels[idx])
                    output_levels.append(result_levels[idx])

                window_data = []
                window_labels = []

            num_samples += 1

        output_counts = np.bincount(output_levels, minlength=self.num_outputs)
        observed_rates = output_counts / num_samples

        return EarlyExitResult(predictions=np.vstack(predictions).reshape(-1),
                               output_levels=np.vstack(output_levels).reshape(-1),
                               labels=np.vstack(labels).reshape(-1),
                               observed_rates=observed_rates,
                               selection_counts=selection_counts,
                               num_changed=num_changed)


class BufferedMaxProb(BufferedExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_max_prob_metric(probs=probs)


class EvenThresholdExiter(ThresholdExiter):

    def __init__(self, epsilon: float, horizon: int, rates: List[float]):
        super().__init__(rates=rates)
        self._epsilon = epsilon
        self._horizon = horizon
        self._stay_counter: Counter = Counter()
        self._elevate_counter: Counter = Counter()

        self._level_window = deque()
        self._window_size = (self._horizon / 2)
        self._rand_rate = 0.5

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def horizon(self) -> int:
        return self._horizon

    def update(self, first_pred: int, pred: int, level: int):
        if level == 0:
            self._stay_counter[pred] += 1
        else:
            self._elevate_counter[pred] += 1
            self._prior[first_pred, pred] += 1

        if self._prev_level != -1:
            self._temporal_counts[level, self._prev_level] += 1

        self._prev_level = level

        # Update the recent window
        self._level_window.append(level)
        while len(self._level_window) > self._window_size:
            self._level_window.popleft()

        # Update the probability adjustments
        total_count = self._stay_counter[pred] + self._elevate_counter[pred]
        stay_count = self._stay_counter[pred]
        elevate_count = self._elevate_counter[pred]

        expected_stay = total_count * self.rates[0]
        expected_elevate = total_count * self.rates[1]

        stay_count = stay_count if total_count > 0 else 0
        elevate_count = elevate_count if total_count > 0 else 0

        stay_diff = expected_stay - stay_count
        elevate_diff = expected_elevate - elevate_count
        width = self.horizon * 2.0

        self._prob_adjustments[0, pred] = linear_step(x=stay_diff, width=width, clip=1.0)
        self._prob_adjustments[1, pred] = linear_step(x=elevate_diff, width=width, clip=1.0)

    def get_prediction(self, probs: np.ndarray, level: int) -> Tuple[int, bool]:
        level_probs = probs[level]  # [K]
        pred = np.argmax(level_probs)

        # Make sure the prediction does not go out of the hard bound
        stay_count = self._stay_counter[pred]
        elevate_count = self._elevate_counter[pred]
        total_count = stay_count + elevate_count + 1  # Account for this sample

        expected_stay = self._rates[0] * total_count
        expected_elevate = self._rates[1] * total_count

        if (level == 0) and (abs(expected_stay - (stay_count + 1)) > self.horizon):
            sorted_preds = np.argsort(level_probs)[::-1]

            for i in range(1, len(sorted_preds)):
                stay_count = self._stay_counter[sorted_preds[i]] + 1
                total_count = stay_count + self._elevate_counter[sorted_preds[i]]
                expected_stay = self._rates[0] * total_count

                if abs(expected_stay - stay_count) <= self.horizon:
                    return sorted_preds[i], True
        elif (level == 1) and (abs(expected_elevate - (elevate_count + 1)) > self.horizon):
            sorted_preds = np.argsort(level_probs)[::-1]

            for i in range(1, len(sorted_preds)):
                elevate_count = self._elevate_counter[sorted_preds[i]] + 1
                total_count = elevate_count + self._stay_counter[sorted_preds[i]]
                expected_elevate = self._rates[1] * total_count

                if abs(expected_elevate - elevate_count) <= self.horizon:
                    return sorted_preds[i], True

        return pred, False

    def reset(self, num_labels: int, pred_rates: np.ndarray):
        self._stay_counter = Counter()
        self._elevate_counter = Counter()
        self._prior = np.eye(num_labels)
        self._prob_adjustments = np.zeros(shape=(2, num_labels))
        self._temporal_counts = np.zeros(shape=(len(self.rates), len(self.rates)))
        self._num_labels = num_labels
        self._prev_level = -1

    def select_output(self, probs: np.ndarray) -> Tuple[int, SelectionType]:
        first_pred = np.argmax(probs[0])
        total_count = self._stay_counter[first_pred] + self._elevate_counter[first_pred] + 1
        stay_rate = (self._stay_counter[first_pred] + 1) / total_count
        stay_cost = abs(stay_rate - self.rates[0])

        # Compute the level from both data-dependent exiting and even-ness exiting
        # to mitigate timing attacks against this conditional behavior
        policy_level, _ = super().select_output(probs=probs)
        even_level = int(stay_rate > self.rates[0])

        all_elevate = np.all(np.isclose(self._prob_adjustments[0], -1.0))
        all_stay = np.all(np.isclose(self._prob_adjustments[0], 1.0))
        even_bound = min((self.horizon - 2.0) / (total_count), self.epsilon)

        r = np.random.uniform()
        use_rand = (r < self._rand_rate)

        obs_elev_rate = np.sum(self._level_window) / self._window_size if len(self._level_window) == self._window_size else self.rates[1]
        rand_level_rate = (1.0 / self._rand_rate) * (self.rates[1] - (1.0 - self._rand_rate) * obs_elev_rate)
        rand_level_rate = max(min(rand_level_rate, 1.0), 0.0)
        #rand_level_rate = self.rates[1]
        rand_level = int(np.random.uniform() < rand_level_rate)

        # Use the consecutive rates to remove temporal correlations
        #temporal_level = 0
        #use_temporal = False

        #if (self._prev_level != -1) and np.all(np.sum(self._temporal_counts, axis=-1) >= self.horizon):
        #    normalized_temporal = self._temporal_counts / np.sum(self._temporal_counts, axis=-1, keepdims=True)
        #    temporal_rates = normalized_temporal[self._prev_level]
        #    margins = np.array([r * (1.0 + self.epsilon) for r in self.rates])

        #    rate_diff = margins - temporal_rates
        #    temporal_level = np.argmax(rate_diff)
        #    use_temporal = np.any(rate_diff < 0)

        #global_bound = np.sqrt(self._window_size * self.rates[0] * self.rates[1])

        if self.rates[0] < SMALL_NUMBER:
            return 1, SelectionType.POLICY
        elif self.rates[1] < SMALL_NUMBER:
            return 0, SelectionType.POLICY
        elif all_elevate:
            return 1, SelectionType.GREEDY
        elif all_stay:
            return 0, SelectionType.GREEDY
        elif use_rand:
            return rand_level, SelectionType.RANDOM
        elif (stay_cost < even_bound):
            return policy_level, SelectionType.POLICY
        else:
            return even_level, SelectionType.GREEDY


class EvenMaxProbExit(EvenThresholdExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_max_prob_metric(probs=probs)


class EvenEntropyExit(EvenThresholdExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_entropy_metric(probs=probs)


class LabelThresholdExiter(EarlyExiter):

    def __init__(self, rates: List[float]):
        super().__init__(rates=rates)
        self._thresholds: Optional[np.ndarray] = None

    @property
    def thresholds(self) -> np.ndarray:
        assert self._thresholds is not None, 'Must call init_thresholds() first'
        return self._thresholds

    def init_thresholds(self, num_labels: int):
        if self._thresholds is not None:
            return

        self._thresholds = np.zeros(shape=(self.num_outputs, num_labels))

    def set_threshold(self, t: float, level: int, label: int):
        assert self._thresholds is not None, 'Must call init_thresholds() first'
        self._thresholds[level, label] = t

    def get_threshold(self, level: int, label: int) -> float:
        assert self._thresholds is not None, 'Must call init_thresholds() first'
        return self._thresholds[level, label]

    def get_thresholds(self, level: int) -> np.ndarray:
        assert self._thresholds is not None, 'Must call init_thresholds() first'
        return self._thresholds[level]

    def get_quantile(self, level: int) -> float:
        return 1.0 - self.rates[level]

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        num_samples, num_outputs, num_labels = val_probs.shape

        assert self.num_outputs == 2, 'Only supports 2 outputs'
        assert num_outputs == self.num_outputs, 'Expected {} outputs. Got {}'.format(self.num_outputs, num_outputs)

        # Initialize the thresholds once we have the true label count
        self.init_thresholds(num_labels=num_labels)

        # Get the maximum probabilities from the first output
        metrics = self.compute_metric(probs=val_probs)  # [B, L]
        first_preds = np.argmax(val_probs[:, 0, :], axis=-1)  # [B]

        # Get the max prob for each prediction of the first output
        pred_distributions: DefaultDict[int, List[float]] = defaultdict(list)
        for sample_idx in range(num_samples):
            pred = first_preds[sample_idx]
            pred_distributions[pred].append(metrics[sample_idx, 0])

        # Set the thresholds according to the percentile in each prediction
        for pred, distribution in pred_distributions.items():
            if np.isclose(self.rates[0], 0.0):
                t = 1.0 + SMALL_NUMBER
            elif np.isclose(self.rates[0], 1.0):
                t = 0.0
            else:
                t = np.quantile(distribution, q=self.get_quantile(level=0))

            self.set_threshold(t=t, level=0, label=pred)
            self.set_threshold(t=0.0, level=1, label=pred)

    def select_output(self, probs: np.ndarray) -> Tuple[int, SelectionType]:
        # Get the threshold
        first_probs = probs[0, :]
        first_pred = int(np.argmax(first_probs))
        t = self.get_threshold(level=0, label=first_pred)

        # Get the metric on the first prediction
        metric = self.compute_metric(first_probs)
        level = int(metric < t)

        return level, SelectionType.POLICY


class LabelMaxProbExit(LabelThresholdExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_max_prob_metric(probs=probs)


class LabelEntropyExit(LabelThresholdExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_entropy_metric(probs=probs)


class RollingExit(LabelThresholdExiter):

    def __init__(self, rates: List[float]):
        super().__init__(rates=rates)
        self._level_queue = deque()
        self._policy_decisions: List[int] = []
        self._window_min = 5
        self._window_max = 10

        self._rand_min = 0.0
        self._rand_max = 1.0

        self._window_size = 0
        self._num_to_elevate = 0
        self._sim_streak = 0
        self._prev_pred = -1

        self._threshold_rates = np.arange(0.0, 1.01, 0.05)

    @property
    def window_min(self) -> int:
        return self._window_min

    @property
    def window_max(self) -> int:
        return self._window_max

    @property
    def rand_min(self) -> float:
        return self._rand_min

    @property
    def rand_max(self) -> float:
        return self._rand_max

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        num_samples, num_outputs, num_labels = val_probs.shape

        assert self.num_outputs == 2, 'Only supports 2 outputs'
        assert num_outputs == self.num_outputs, 'Expected {} outputs. Got {}'.format(self.num_outputs, num_outputs)

        # Initialize the thresholds once we have the true label count
        self._thresholds = np.zeros(shape=(num_labels, len(self._threshold_rates)))

        # Get the maximum probabilities from the first output
        metrics = self.compute_metric(probs=val_probs)  # [B, L]
        first_preds = np.argmax(val_probs[:, 0, :], axis=-1)  # [B]

        # Get the max prob for each prediction of the first output
        pred_distributions: DefaultDict[int, List[float]] = defaultdict(list)
        for sample_idx in range(num_samples):
            pred = first_preds[sample_idx]
            pred_distributions[pred].append(metrics[sample_idx, 0])

        # Set the thresholds according to the percentile in each prediction
        for rate_idx, rate in enumerate(self._threshold_rates):
            for pred, distribution in pred_distributions.items():
                if np.isclose(rate, 0.0):
                    t = 1.0 + SMALL_NUMBER
                elif np.isclose(rate, 1.0):
                    t = 0.0
                else:
                    t = np.quantile(distribution, q=(1.0 - rate))

            self._thresholds[pred, rate_idx] = t

    def reset(self, num_labels: int, pred_rates: np.ndarray):
        self._level_queue = deque()
        self._policy_decisions = []
        #self._sim_streak = 0
        self._prev_pred = -1
        self._window_size = np.random.randint(low=self.window_min + 1, high=self.window_max + 1)

        int_part = int(self.rates[1] * self._window_size)
        frac_part = (self.rates[1] * self._window_size) - int_part
        self._num_to_elevate = int_part + int(np.random.uniform() < frac_part)

    def select_output(self, probs: np.ndarray) -> Tuple[int, SelectionType]:
        # TODO: Remove conditional logic to avoid timing attacks
        if len(self._level_queue) == self._window_size:
            self.reset(num_labels=0, pred_rates=None)
            #print('==========')

        # Get the level from random exiting
        num_remaining = self._window_size - len(self._level_queue)
        level_sum = sum(self._level_queue) if len(self._level_queue) > 0 else 0
        remaining_to_elevate = self._num_to_elevate - level_sum
        rand_elev_rate = remaining_to_elevate / num_remaining
        rand_level = int(np.random.uniform() < rand_elev_rate)

        # Get the threshold index based on the randomness elevation rate
        target_exit_rate = rand_elev_rate
        threshold_idx = np.argmin(np.abs(self._threshold_rates - target_exit_rate))

        metrics = self.compute_metric(probs)
        first_metric = metrics[0]
        pred = np.argmax(probs[0], axis=-1)
        data_level = int(first_metric < self._thresholds[pred, threshold_idx])
        self._policy_decisions.append(data_level)

        r = np.random.uniform()

        policy_num_elevate = sum(self._policy_decisions)
        expected_diff = abs((self.rates[1] * len(self._policy_decisions)) - policy_num_elevate)
        #rand_rate = (self.rand_max - self.rand_min) * (1.0 - np.power(2.0, -1 * expected_diff)) + self.rand_min
        rand_rate = 0.5

        if remaining_to_elevate == 0:
            level = 0
            selection = SelectionType.GREEDY
        elif num_remaining == remaining_to_elevate:
            level = 1
            selection = SelectionType.GREEDY
        elif r < rand_rate:
            level = rand_level
            selection = SelectionType.RANDOM
        else:
            level = data_level
            selection = SelectionType.POLICY

        self._prev_pred = pred
        #print('Level: {}, Selection Type: {}, Rand Elev Rate: {:.5f}, Rand Rate: {}'.format(level, selection, rand_elev_rate, rand_rate))

        self._level_queue.append(level)
        return level, selection


class RollingMaxProb(RollingExit):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_max_prob_metric(probs=probs)


class AdaptiveRandomExit(LabelThresholdExiter):

    def __init__(self, epsilon: float, rates: List[float]):
        super().__init__(rates=rates)
        assert epsilon >= 0.0 and epsilon <= 1.0, 'Epsilon must be in [0, 1]'
        assert len(rates) == 2, 'Adaptive Random Exiting only works with 2-level models.'

        self._epsilon = epsilon

        self._horizon_min = 5
        self._horizon_max = 15

        self._fwd_horizon = 0
        self._bwd_horizon = 0

        self._adaptive_elevation_rate = self.rates[1]
        self._rand_rate = 0.5
        self._rand_counter_limit = 5
        self._rand_counter = 0
        
        self._level_queue = deque()

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def window_size(self) -> int:
        return self._window_size

    def reset(self, num_labels: int, pred_rates: np.ndarray):
        self._fwd_horizon = 0
        self._bwd_horizon = 0
        self._rand_rate = 0.5
        self._rand_counter = 0
        self._adaptive_elevation_rate = self.rates[1]
        self._level_queue = deque()

    def select_output(self, probs: np.ndarray) -> Tuple[int, SelectionType]:
        metrics = self.compute_metric(probs)
        first_metric = metrics[0]
        first_pred = np.argmax(probs[0])
        level = int(first_metric < self.get_threshold(label=first_pred, level=0))

        first_metric = metrics[0]
        pred = np.argmax(probs[0])

        # Get the forward and backward horizion values via random sampling
        self._fwd_horizon = np.random.randint(low=self._horizon_min, high=self._horizon_max + 1, dtype=int)
        self._bwd_horizon = np.random.randint(low=self._horizon_min, high=self._horizon_max + 1, dtype=int)

        # Set the randomness rate based on the difference between observed
        # and expected elevation rates
        bwd_decisions = list(self._level_queue)[0:self._bwd_horizon]
        num_elevated = sum(bwd_decisions) if len(bwd_decisions) > 0 else 0
        expected_elevated = self.rates[1] * len(bwd_decisions)
        expected_diff = abs(expected_elevated - num_elevated) / 2.0

        #rand_rate = 1.0 - np.power(2.0, -1 * expected_diff)
        rand_rate = 0.25

        # Set the randomness elevation fraction based on the observed
        # rates over the previous window
        elevation_quota = self.rates[1] * (len(bwd_decisions) + self._fwd_horizon)
        remaining_to_elevate = max(elevation_quota - num_elevated, 0)
        adaptive_elevation_rate = remaining_to_elevate / self._fwd_horizon

        self._rand_counter += 1

        if self._rand_counter == self._rand_counter_limit:
            self._rand_rate = rand_rate
            self._adaptive_elevation_rate = adaptive_elevation_rate
            self._rand_counter = 0

        should_use_random = (np.random.uniform() < self._rand_rate)
        rand_level = int(np.random.uniform() < self._adaptive_elevation_rate)

        # Add the data-dependent decision to the level queue
        self._level_queue.append(level)
        while len(self._level_queue) > self._horizon_max:
            self._level_queue.popleft()

        if should_use_random:
            return rand_level, SelectionType.RANDOM
        else:
            return level, SelectionType.POLICY


class AdaptiveRandomMaxProb(AdaptiveRandomExit):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_max_prob_metric(probs=probs)


class EvenLabelThresholdExiter(LabelThresholdExiter):

    def __init__(self, epsilon: float, horizon: int, rates: List[float]):
        super().__init__(rates=rates)
        self._epsilon = epsilon
        self._horizon = horizon
        self._stay_counter: Counter = Counter()
        self._elevate_counter: Counter = Counter()

        self._level_window = deque()
        self._window_size = 25

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def horizon(self) -> int:
        return self._horizon

    def update(self, first_pred: int, pred: int, level: int):
        if level == 0:
            self._stay_counter[pred] += 1
        else:
            self._elevate_counter[pred] += 1
            self._prior[first_pred, pred] += 1

        # Update the recent window
        self._level_window.append(level)
        while len(self._level_window) > self._window_size:
            self._level_window.popleft()

        for pred in range(self._num_labels):
            total_count = self._stay_counter[pred] + self._elevate_counter[pred]

            stay_count = self._stay_counter[pred]
            elevate_count = self._elevate_counter[pred]

            expected_stay = total_count * self.rates[0]
            expected_elevate = total_count * self.rates[1]

            stay_count = stay_count if total_count > 0 else 0
            elevate_count = elevate_count if total_count > 0 else 0

            stay_diff = expected_stay - stay_count
            elevate_diff = expected_elevate - elevate_count
            width = self.horizon * 2.0

            self._prob_adjustments[0, pred] = linear_step(x=stay_diff, width=width, clip=1.0)
            self._prob_adjustments[1, pred] = linear_step(x=elevate_diff, width=width, clip=1.0)

    def get_prediction(self, probs: np.ndarray, level: int) -> int:
        level_probs = probs[level]  # [K]
        adjusted_probs = level_probs + mask_non_max(self._prob_adjustments[level])  # [K]
        return np.argmax(adjusted_probs)

    def reset(self, num_labels: int, pred_rates: np.ndarray):
        self._stay_counter = Counter()
        self._elevate_counter = Counter()
        self._prior = np.eye(num_labels)
        self._prob_adjustments = np.zeros(shape=(2, num_labels))
        self._num_labels = num_labels

    def select_output(self, probs: np.ndarray) -> int:
        adjusted_probs = mask_non_max(self._prob_adjustments) + probs

        first_pred = np.argmax(adjusted_probs[0])
        total_count = self._stay_counter[first_pred] + self._elevate_counter[first_pred] + 1
        stay_rate = (self._stay_counter[first_pred] + 1) / total_count
        stay_cost = abs(stay_rate - self.rates[0])

        # Compute the level from both data-dependent exiting and even-ness exiting
        # to avoid timing attacks against this conditional behavior
        policy_level = super().select_output(probs=probs)
        even_level = int(stay_rate > self.rates[0])

        all_elevate = np.all(np.isclose(self._prob_adjustments[0], -1.0))
        all_stay = np.all(np.isclose(self._prob_adjustments[0], 1.0))
        even_bound = min(max((self.horizon - 1.0), self.epsilon * total_count) / total_count, self.epsilon)

        if self.rates[0] < SMALL_NUMBER:
            return 1
        elif self.rates[1] < SMALL_NUMBER:
            return 0
        elif all_elevate:
            return 1
        elif all_stay:
            return 0
        elif (stay_cost < even_bound):
            return policy_level
        else:
            return even_level


class EvenLabelMaxProbExit(EvenLabelThresholdExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_max_prob_metric(probs=probs)


class EvenLabelEntropyExit(EvenLabelThresholdExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_entropy_metric(probs=probs)


class HybridRandomExit(LabelThresholdExiter):

    def __init__(self, rates: List[float], path: Optional[str], metric_name: str):
        super().__init__(rates=rates)
        assert metric_name in ('max-prob', 'entropy'), 'Metric name must be `max-prob` or `entropy`'

        self._thresholds: Optional[np.ndarray] = None
        self._rand_rate: Optional[np.ndarray] = None
        self._observed_rates: Optional[np.ndarray] = None
        self._trials = 1
        self._metric_name = metric_name

        self._noise_scale = 0.02
        self._epsilon = 0.001

        if path is not None:
            folder, file_name = os.path.split(path)
            model_name = file_name.split('.')[0]
            thresholds_path = os.path.join(folder, '{}_{}-thresholds.json'.format(model_name, metric_name))

            saved_thresholds = read_json(thresholds_path)
            key = str(round(rates[0], 2))
            self._thresholds = np.array(saved_thresholds['thresholds'][key])
            self._rand_rate = np.array(saved_thresholds['rand_rate'][key])
            self._weights = np.array(saved_thresholds['weights'][key])

    @property
    def metric_name(self) -> str:
        return self._metric_name

    def compute_metric(self, probs: np.ndarray) -> float:
        raise NotImplementedError()

    def select_output(self, probs: np.ndarray) -> int:
        if self._rand_rate is None:
            return super().select_output(probs)

        # Get the maximum probability and prediction on the first output
        first_probs = probs[0, :]
        first_pred = int(np.argmax(first_probs))
        metric = self.compute_metric(first_probs)

        # Get the threshold and scaling weight
        t = self.get_threshold(level=0, label=first_pred)
        w = np.square(self._weights[first_pred])

        # Compute the elevation probability
        level_prob = sigmoid(w * (t - metric))
        level = int(np.random.uniform() < level_prob)

        # Determine whether we should act randomly
        should_act_random = int(np.random.uniform() < rand_rate)
        rand_level = int(np.random.uniform() < self.rates[1])

        return should_act_random * rand_level + (1 - should_act_random) * level

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        if self._thresholds is not None:
            return

        super().fit(val_probs=val_probs, val_labels=val_labels)

        best_loss = BIG_NUMBER
        best_thresholds = self.thresholds[0]
        best_weights = np.ones_like(best_thresholds)
        best_rates = np.ones_like(best_thresholds)
        learning_rates = [1e-2]
        target = 1.0 - self.rates[0]

        rand = np.random.RandomState(seed=2890)

        val_preds = np.argmax(val_probs, axis=-1)
        is_correct = np.equal(val_preds, np.expand_dims(val_labels, axis=-1)).astype(float)

        for lr in learning_rates:
            for trial in range(self._trials):
                start_thresholds = np.copy(self.thresholds[0])
                start_weights = rand.uniform(low=1.0, high=2.0, size=start_thresholds.shape)

                if trial > 0:
                    start_thresholds += rand.uniform(low=-1 * self._noise_scale,
                                                           high=self._noise_scale,
                                                           size=start_thresholds.shape)

                loss, thresholds, weights, rates = fit_thresholds_grad(probs=val_probs[:, 0, :],
                                                                       preds=val_preds,
                                                                       target=target,
                                                                       start_thresholds=start_thresholds,
                                                                       start_weights=start_weights,
                                                                       train_thresholds=True,
                                                                       train_weights=True,
                                                                       learning_rate=lr,
                                                                       metric_name=self.metric_name)

                # Fit the sharpness parameter
                _, weights, rates = fit_sharpness(probs=val_probs[:, 0, :],
                                                  labels=val_labels,
                                                  target=target,
                                                  epsilon=self._epsilon,
                                                  start_weights=weights,
                                                  thresholds=thresholds,
                                                  metric_name=self.metric_name,
                                                  learning_rate=1e-4)

                if loss < best_loss:
                    best_loss = loss
                    best_thresholds = thresholds
                    best_weights = weights
                    best_rates = rates

                # If we get to 0 loss, no need to continue
                if best_loss < SMALL_NUMBER:
                    break

            # If we get to 0 loss, no need to continue
            if best_loss < SMALL_NUMBER:
                break

        print('==========')
        print('Exit Rates: {}'.format(best_rates))

        rand_rate = fit_randomization(avg_rates=best_rates,
                                      labels=val_labels,
                                      epsilon=self._epsilon,
                                      target=target)

        print('Randomness Rate: {}'.format(rand_rate))
        print('Weights: {}'.format(best_weights))

        #rand_rates = fit_randomization(probs=val_probs[:, 0, :],
        #                               labels=val_labels,
        #                               thresholds=best_thresholds,
        #                               target=target,
        #                               epsilon=self._epsilon,
        #                               metric_name=self.metric_name)

        self._thresholds[0] = best_thresholds
        self._rand_rate = rand_rate
        self._weights = best_weights


class HybridMaxProbExit(HybridRandomExit):

    def __init__(self, rates: List[float], path: Optional[str]):
        super().__init__(rates=rates, path=path, metric_name='max-prob')

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_max_prob_metric(probs=probs)


class HybridEntropyExit(HybridRandomExit):

    def __init__(self, rates: List[float], path: Optional[str]):
        super().__init__(rates=rates, path=path, metric_name='entropy')

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_entropy_metric(probs=probs)


def make_policy(strategy: ExitStrategy, rates: List[float], model_path: str) -> EarlyExiter:
    if strategy == ExitStrategy.RANDOM:
        return RandomExit(rates=rates)
    elif strategy == ExitStrategy.ENTROPY:
        return EntropyExit(rates=rates)
    elif strategy == ExitStrategy.MAX_PROB:
        return MaxProbExit(rates=rates)
    elif strategy == ExitStrategy.LABEL_MAX_PROB:
        return LabelMaxProbExit(rates=rates)
    elif strategy == ExitStrategy.LABEL_ENTROPY:
        return LabelEntropyExit(rates=rates)
    elif strategy == ExitStrategy.HYBRID_MAX_PROB:
        return HybridMaxProbExit(rates=rates, path=model_path)
    elif strategy == ExitStrategy.HYBRID_ENTROPY:
        return HybridEntropyExit(rates=rates, path=model_path)
    elif strategy == ExitStrategy.GREEDY_EVEN:
        return GreedyEvenExit(rates=rates)
    elif strategy == ExitStrategy.EVEN_MAX_PROB:
        return EvenMaxProbExit(rates=rates, epsilon=0.01, horizon=10)
    elif strategy == ExitStrategy.EVEN_LABEL_MAX_PROB:
        return EvenLabelMaxProbExit(rates=rates, epsilon=0.01, horizon=10)
    elif strategy == ExitStrategy.BUFFERED_MAX_PROB:
        return BufferedMaxProb(rates=rates, window_size=10, epsilon=0.02, pred_window=50)
    elif strategy == ExitStrategy.DELAYED_MAX_PROB:
        return DelayedMaxProb(rates=rates, window_size=10, delay_prob=0.5)
    elif strategy == ExitStrategy.ADAPTIVE_RANDOM_MAX_PROB:
        return AdaptiveRandomMaxProb(rates=rates, epsilon=0.05)
    elif strategy == ExitStrategy.ROLLING_MAX_PROB:
        return RollingMaxProb(rates=rates)
    else:
        raise ValueError('No policy {}'.format(strategy))
