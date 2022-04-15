import numpy as np
import time
from annoy import AnnoyIndex
from collections import namedtuple, defaultdict, Counter
from enum import Enum, auto
from typing import Any, List, Tuple, Dict, DefaultDict, Optional

from privddnn.dataset.data_iterators import DataIterator
from privddnn.utils.metrics import compute_max_prob_metric, compute_entropy_metric, sigmoid, compute_entropy
from privddnn.utils.constants import BIG_NUMBER, SMALL_NUMBER
from privddnn.utils.exit_utils import get_adaptive_elevation_bounds, normalize_exit_rates
from privddnn.utils.random import RandomUniformGenerator, RandomChoiceGenerator, RandomIntGenerator
from .linear_optimizer import GeneticLinearOptimizer


EarlyExitResult = namedtuple('EarlyExitResult', ['predictions', 'output_levels', 'labels', 'observed_rates', 'monitor_stats'])
BufferedEntry = namedtuple('BufferedEntry', ['probs', 'label'])
BufferedResult = namedtuple('BufferedResult', ['preds', 'exit_decisions', 'labels'])


class ExitStrategy(Enum):
    RANDOM = auto()
    ENTROPY = auto()
    MAX_PROB = auto()
    LABEL_ENTROPY = auto()
    LABEL_MAX_PROB = auto()
    BUFFERED_MAX_PROB = auto()
    BUFFERED_ENTROPY = auto()
    ADAPTIVE_RANDOM_MAX_PROB = auto()
    ADAPTIVE_RANDOM_ENTROPY = auto()


class EarlyExiter:

    def __init__(self, rates: List[float]):
        assert np.isclose(np.sum(rates), 1.0), 'Rates must sum to 1'
        self._rates = rates
        self._num_labels = -1

    @property
    def rates(self) -> List[float]:
        return self._rates

    @property
    def num_outputs(self) -> int:
        return len(self._rates)

    @property
    def num_labels(self) -> int:
        return self._num_labels

    @property
    def name(self) -> str:
        raise NotImplementedError()

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        self._num_labels = val_probs.shape[-1]

    def select_output(self, probs: np.ndarray) -> int:
        raise NotImplementedError()

    def reset(self):
        pass

    def get_prediction(self, probs: np.ndarray, level: int) -> int:
        return np.argmax(probs[level])

    def init_monitor_dict(self) -> Dict[str, List[float]]:
        return dict()

    def record_monitor_step(self) -> Dict[str, float]:
        return dict()

    def test(self, data_iterator: DataIterator,
                   max_num_samples: Optional[int]) -> EarlyExitResult:
        predictions: List[int] = []
        output_levels: List[int] = []
        labels: List[int] = []

        self.reset()
        monitor_dict = self.init_monitor_dict()

        num_changed = 0
        num_samples = 0

        for _, sample_probs, label in data_iterator:
            if (max_num_samples is not None) and (num_samples >= max_num_samples):
                break

            level = self.select_output(probs=sample_probs)
            pred = self.get_prediction(probs=sample_probs, level=level)

            monitor_step = self.record_monitor_step()
            for key, value in monitor_step.items():
                monitor_dict[key].append(value)

            predictions.append(pred)
            output_levels.append(level)
            labels.append(label)
            num_samples += 1

        output_counts = np.bincount(output_levels, minlength=self.num_outputs)
        observed_rates = output_counts / num_samples

        return EarlyExitResult(predictions=np.vstack(predictions).reshape(-1),
                               output_levels=np.vstack(output_levels).reshape(-1),
                               labels=np.vstack(labels).reshape(-1),
                               observed_rates=observed_rates,
                               monitor_stats=monitor_dict)


class RandomExit(EarlyExiter):

    def __init__(self, rates: List[float]):
        super().__init__(rates=rates)
        self._random_choice = RandomChoiceGenerator(num_choices=len(rates),
                                                    rates=rates,
                                                    batch_size=5000)

    @property
    def name(self) -> str:
        return 'random'

    def select_output(self, probs: np.ndarray) -> int:
        level = self._random_choice.get()
        return int(level)


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

    def select_output(self, probs: np.ndarray) -> int:
        metric = self.compute_metric(probs)
        comp = np.greater(metric, self.thresholds).astype(int)  # [L]
        return np.argmax(comp)  # Breaks ties by selecting the first value

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        super().fit(val_probs=val_probs, val_labels=val_labels)

        assert val_probs.shape[0] == val_labels.shape[0], 'Misaligned probabilites and labels'
        assert val_probs.shape[1] == self.num_outputs, 'Expected {} outputs. Got {}'.format(self.num_outputs, val_probs.shape[1])
        assert val_probs.shape[2] == self.num_labels, 'Expected {} labels. Got {}'.format(self.num_labels, val_probs.shape[2])

        # Compute the maximum probability for each predicted distribution
        metrics = self.compute_metric(probs=val_probs)  # [B, L]

        # Comute the thresholds based on the quantile
        for level in range(self.num_outputs):
            rate_sum = sum(self.rates[level:])
            level_rate = self.rates[level] / rate_sum if rate_sum > 0.0 else 0.0

            if np.isclose(level_rate, 1.0):
                t = 0.0
            elif np.isclose(level_rate, 0.0):
                t = 1.0
            else:
                t = np.quantile(metrics[:, level], q=(1.0 - level_rate))

            # Mask out the elements stopped by the current threshold
            remaining_idx = [i for i in range(len(metrics)) if metrics[i, level] < t]
            metrics = metrics[remaining_idx]

            # Set the threshold for this exit point
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

    def __init__(self, rates: List[float], window_size: int):
        super().__init__(rates=rates)
        self._window_size = window_size

    @property
    def window_size(self) -> int:
        return self._window_size

    def get_exit_quotas(self, window_size: int) -> List[int]:
        result: List[int] = []

        for rate in self.rates:
            if np.isclose(rate, 0.0):
                exit_quota = 0
            elif np.isclose(rate, 1.0):
                exit_quota = window_size
            else:
                int_part = int(rate * window_size)
                frac_part = (rate * window_size) - int_part
                rand_add = int(np.random.uniform() < frac_part)
                exit_quota = int_part + rand_add

            result.append(exit_quota)

        return result

    def process_window(self, window_probs: [np.ndarray], window_labels: List[int]) -> BufferedResult:
        window_size = len(window_probs)
        exit_quotas = self.get_exit_quotas(window_size=window_size)

        probs = np.vstack([np.expand_dims(arr, axis=0) for arr in window_probs])  # [W, L, C]

        exit_decisions = np.zeros(shape=(window_size, ))  # [W]
        preds = np.zeros(shape=(window_size, ))  # [W]
        labels = np.zeros(shape=(window_size, ))  # [W]

        used_indices: Set[int] = set()

        for level in range(len(self.rates)):
            # Compute all of the metrics for this level
            level_probs = probs[:, level, :]
            level_preds = np.argmax(level_probs, axis=-1)
            level_metrics = self.compute_metric(level_probs)  # [W]

            # Get the quota for this exit point
            level_quota = exit_quotas[level]
            if level_quota == 0:
                continue

            # Get the elements with the highest confidence metric values (filter out used entries)
            sorted_indices = np.argsort(level_metrics)[::-1]  # [W] sorted indices from highest to lowest
            selected_indices = list(filter(lambda idx: (idx not in used_indices), sorted_indices))[0:level_quota]
            used_indices.update(selected_indices)

            for idx in selected_indices:
                exit_decisions[idx] = level
                preds[idx] = level_preds[idx]
                labels[idx] = window_labels[idx]

        # Shuffle the result
        sample_idx = np.arange(window_size)
        np.random.shuffle(sample_idx)

        return BufferedResult(exit_decisions=exit_decisions[sample_idx].astype(int).tolist(),
                              preds=preds[sample_idx].astype(int).tolist(),
                              labels=labels[sample_idx].astype(int).tolist())

    def test(self, data_iterator: DataIterator, max_num_samples: Optional[int]) -> EarlyExitResult:
        predictions: List[int] = []
        output_levels: List[int] = []
        labels: List[int] = []

        window_probs: List[np.ndarray] = []
        window_labels: List[int] = []

        self.reset()

        elev_count = int(self.window_size * self.rates[1])
        elev_remainder = (self.window_size * self.rates[1]) - elev_count

        num_samples = 0
        for _, sample_probs, label in data_iterator:
            if (max_num_samples is not None) and (num_samples >= max_num_samples):
                break

            # Add the entry to the current window
            window_probs.append(sample_probs)
            window_labels.append(label)

            if len(window_probs) == self.window_size:
                buffered_result = self.process_window(window_probs=window_probs,
                                                      window_labels=window_labels)

                predictions.extend(buffered_result.preds)
                output_levels.extend(buffered_result.exit_decisions)
                labels.extend(buffered_result.labels)

                window_probs = []
                window_labels = []

            num_samples += 1

        if len(window_probs) > 0:
            buffered_result = self.process_window(window_probs=window_probs,
                                                  window_labels=window_labels)

            predictions.extend(buffered_result.preds)
            output_levels.extend(buffered_result.exit_decisions)
            labels.extend(buffered_result.labels)

        output_counts = np.bincount(output_levels, minlength=self.num_outputs)
        observed_rates = output_counts / num_samples

        return EarlyExitResult(predictions=np.vstack(predictions).reshape(-1),
                               output_levels=np.vstack(output_levels).reshape(-1),
                               labels=np.vstack(labels).reshape(-1),
                               observed_rates=observed_rates,
                               monitor_stats=dict())


class BufferedMaxProb(BufferedExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_max_prob_metric(probs=probs)


class BufferedEntropy(BufferedExiter):

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

    def init_thresholds(self):
        if self._thresholds is not None:
            return

        self._thresholds = np.zeros(shape=(self.num_outputs, self.num_labels))

    def set_threshold(self, t: float, level: int, pred: int):
        assert self._thresholds is not None, 'Must call init_thresholds() first'
        self._thresholds[level, pred] = t

    def get_threshold(self, level: int, pred: int) -> float:
        assert self._thresholds is not None, 'Must call init_thresholds() first'
        return float(self._thresholds[level, pred])

    def get_thresholds(self, level: int) -> np.ndarray:
        assert self._thresholds is not None, 'Must call init_thresholds() first'
        return self._thresholds[level]

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        super().fit(val_probs=val_probs, val_labels=val_labels)
        num_samples, num_outputs, num_labels = val_probs.shape

        assert num_outputs == self.num_outputs, 'Expected {} outputs. Got {}'.format(self.num_outputs, num_outputs)
        assert num_labels == self.num_labels, 'Expected {} labels. Got {}'.format(self.num_labels, num_labels)

        # Initialize the thresholds once we have the true label count
        self.init_thresholds()

        # Compute the confidence metrics and predictions
        metrics = self.compute_metric(probs=val_probs)  # [B, L]
        preds = np.argmax(val_probs, axis=-1)  # [B, L]

        # Set the thresholds according to the percentile in each prediction
        for level in range(self.num_outputs - 1):
            # Stratify the predictions by distribution
            pred_distributions: DefaultDict[int, List[float]] = defaultdict(list)
            for sample_idx in range(len(metrics)):
                pred = preds[sample_idx, level]
                pred_distributions[pred].append(metrics[sample_idx, level])

            # Get the percentile for this level
            rate_sum = sum(self.rates[level:])
            level_rate = self.rates[level] / rate_sum if rate_sum > 0.0 else 0.0

            # Compute the threshold based on the stratified distribution
            for pred, distribution in pred_distributions.items():
                if np.isclose(level_rate, 0.0):
                    t = 1.0 + SMALL_NUMBER
                elif np.isclose(level_rate, 1.0):
                    t = -1 * SMALL_NUMBER
                else:
                    t = np.quantile(distribution, q=(1.0 - level_rate))

                self.set_threshold(t=t, level=level, pred=pred)

            # Remove the samples which exit at this level
            remaining_idx = [i for i in range(len(metrics)) if (metrics[i, level] < self.get_threshold(level=level, pred=preds[i, level]))]
            metrics = metrics[remaining_idx]
            preds = preds[remaining_idx]

        for pred in range(self.num_labels):
            self.set_threshold(t=0.0, level=self.num_outputs - 1, pred=pred)

    def select_output(self, probs: np.ndarray) -> int:
        for level in range(self.num_outputs):
            # Get the threshold for this exit point
            level_probs = probs[level, :]
            level_pred = int(np.argmax(level_probs))
            level_threshold = self.get_threshold(level=level, pred=level_pred)

            # Compare the thresholds to the metric for this level
            level_metric = self.compute_metric(level_probs)

            if level_metric >= level_threshold:
                return level

        return self.num_outputs - 1


class LabelMaxProbExit(LabelThresholdExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_max_prob_metric(probs=probs)


class LabelEntropyExit(LabelThresholdExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_entropy_metric(probs=probs)


class AdaptiveRandomExit(LabelThresholdExiter):

    def __init__(self, rates: List[float]):
        super().__init__(rates=rates)

        self._max_epsilon = 0.5
        self._increase_factor = 2.0
        self._decrease_factor = 0.9
        self._window_min = 5
        self._window_max = 20

        self._rand_uniform = RandomUniformGenerator(batch_size=5000)
        self._rand_choice = RandomChoiceGenerator(num_choices=len(rates), rates=rates, batch_size=5000)
        self._rand_int = RandomIntGenerator(low=self._window_min, high=self._window_max + 1, batch_size=5000)

        self._epsilons = [self._max_epsilon for _ in range(self.num_outputs)]
        self._level_counter: Counter = Counter()
        self._prev_preds: List[int] = []
        self._step = 0

        self._level_rates = normalize_exit_rates(rates=rates)

    @property
    def epsilons(self) -> List[float]:
        return self._epsilons

    def get_epsilon(self, level: int) -> float:
        return self.epsilons[level]

    def increase_epsilon(self, level: int):
        self._epsilons[level] = min(self.epsilons[level] * self._increase_factor, self._max_epsilon)

    def decrease_epsilon(self, level: int):
        self._epsilons[level] *= self._decrease_factor

    def make_level_targets(self) -> List[int]:
        level_targets = [int(self.rates[level] * self._window) for level in range(self.num_outputs)]
        frac_parts = np.array([(self.rates[level] * self._window) - level_targets[level] for level in range(self.num_outputs)])
        sample_probs = frac_parts / max(np.sum(frac_parts), SMALL_NUMBER)

        level_idx = np.arange(self.num_outputs)
        np.random.shuffle(level_idx)

        for level in level_idx:
            if sum(level_targets) == self._window:
                break

            frac_part = (self.rates[level] * self._window) - level_targets[level]
            if self._rand_uniform.get() < frac_part:
                level_targets[level] += 1

        for _ in range(self._window - sum(level_targets)):
            rand_level = int(self._rand_choice.get())
            level_targets[rand_level] += 1

        assert sum(level_targets) == self._window, 'Found an allocation of {}, Expected {}'.format(sum(level_targets), self._window)
        return level_targets

    def reset(self):
        self._window = np.random.randint(low=self._window_min, high=self._window_max + 1)
        self._epsilon = self._max_epsilon
        self._prev_preds: List[int] = [-1 for _ in range(self.num_labels)]

        self._level_counter: Counter = Counter()
        self._level_targets = self.make_level_targets()
        self._step = 0

    def init_monitor_dict(self) -> Dict[str, List[Any]]:
        return {
            'prob_bias': []
        }

    def record_monitor_step(self) -> Dict[str, Any]:
        return {
            'prob_bias': self.epsilons
        }

    def select_output(self, probs: np.ndarray) -> int:
        """
        Selects the exit point to use given the predicted probabilities.

        Args:
            probs: A [L, K] array where L is the number of exit points and K
                is the number of classes.
        """
        metrics = self.compute_metric(probs)
        preds = np.argmax(probs, axis=-1)  # [L]

        level_idx = list(range(self.num_outputs))
        selected_level = self.num_outputs - 1

        for level in range(self.num_outputs):
            # Get the data-dependent information for this level
            level_metric = metrics[level]
            level_probs = probs[level]
            level_pred = preds[level]
            level_threshold = self.get_threshold(level=level, pred=level_pred)

            # Get the exit rate for this output
            level_rate = self._level_rates[level]

            # Get the remaining number of elements to stop at this level
            num_remaining = self._window - self._step  # Number of remaining elements in this window
            remaining_to_exit = self._level_targets[level] - self._level_counter[level]  # Quota of elements that should continue on

            if level_pred == self._prev_preds[level]:
                self.decrease_epsilon(level=level)
            else:
                self.increase_epsilon(level=level)

            min_rate, max_rate = get_adaptive_elevation_bounds(continue_rate=(1.0 - level_rate),
                                                               epsilon=self.get_epsilon(level=level))

            if abs(level_rate - 1.0) < SMALL_NUMBER:
                continue_prob = 0.0
            elif abs(level_rate) < SMALL_NUMBER:
                continue_prob = 1.0
            elif level_metric < level_threshold:
                continue_prob = max_rate
            else:
                continue_prob = min_rate

            should_continue = (self._rand_uniform.get() < continue_prob)
        
            if remaining_to_exit == 0:
                should_continue = True
            elif remaining_to_exit == num_remaining:
                should_continue = False

            self._prev_preds[level] = level_pred

            if (not should_continue):
                self._level_counter[level] += 1
                selected_level = level
                break

        self._step += 1
        if self._step == self._window:
            self._window = self._rand_int.get()
            self._level_counter: Counter = Counter()
            self._level_targets = self.make_level_targets()
            self._step = 0

        return level


class AdaptiveRandomMaxProb(AdaptiveRandomExit):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_max_prob_metric(probs=probs)


class AdaptiveRandomEntropy(AdaptiveRandomExit):

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
    elif strategy == ExitStrategy.BUFFERED_MAX_PROB:
        return BufferedMaxProb(rates=rates, window_size=10)
    elif strategy == ExitStrategy.BUFFERED_ENTROPY:
        return BufferedEntropy(rates=rates, window_size=10)
    elif strategy == ExitStrategy.ADAPTIVE_RANDOM_MAX_PROB:
        return AdaptiveRandomMaxProb(rates=rates)
    elif strategy == ExitStrategy.ADAPTIVE_RANDOM_ENTROPY:
        return AdaptiveRandomEntropy(rates=rates)
    else:
        raise ValueError('No policy {}'.format(strategy))
