import numpy as np
from collections import namedtuple, defaultdict
from enum import Enum, auto
from sklearn.ensemble import AdaBoostClassifier
from typing import List, Tuple, Dict, DefaultDict

from privddnn.utils.metrics import compute_entropy
from privddnn.utils.constants import BIG_NUMBER


EarlyExitResult = namedtuple('EarlyExitResult', ['predictions', 'output_levels', 'observed_rates'])


class EvenMode(Enum):
    PLAIN = auto()
    TOTAL_RAND = auto()
    CLASS_RAND = auto()


class ExitStrategy(Enum):
    RANDOM = auto()
    ENTROPY = auto()
    MAX_PROB = auto()
    LABEL_ENTROPY = auto()
    LABEL_MAX_PROB = auto()


def validate_args(probs: np.ndarray, rates: List[float]):
    num_outputs = probs.shape[1]
    assert len(rates) == num_outputs, 'Must provide {} rates. Got: {}'.format(len(rates), num_outputs)
    assert np.isclose(np.sum(rates), 1.0), 'Rates must sum to 1'


def make_attack_dataset(outputs: np.ndarray, labels: np.ndarray, window_size: int, num_samples: int, rand: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    output_dist: DefaultDict[int, List[int]] = defaultdict(list)

    for num_outputs, label in zip(outputs, labels):
        output_dist[label].append(num_outputs)

    num_labels = len(output_dist)
    samples_per_label = int(num_samples / num_labels)

    input_list: List[np.ndarray] = []
    output_list: List[np.ndarray] = []

    for label, output_counts in output_dist.items():
        for _ in range(samples_per_label):
            selected_counts = rand.choice(output_counts, size=window_size)

            # Create the features
            mean = np.average(selected_counts)
            std = np.std(selected_counts)
            median = np.median(selected_counts)

            input_list.append(np.expand_dims([mean, std, median], axis=0))
            output_list.append(label)

    return np.vstack(input_list), np.vstack(output_list)


class EarlyExiter:

    def __init__(self, rates: List[float]):
        assert np.isclose(np.sum(rates), 1.0), 'Rates must sum to 1'

        self._attack_model = AdaBoostClassifier()
        self._val_rand = np.random.RandomState(seed=28116)
        self._test_rand = np.random.RandomState(seed=52190)
        self._rates = rates

    @property
    def rates(self) -> List[float]:
        return self._rates

    @property
    def num_outputs(self) -> int:
        return len(self._rates)

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        pass

    def select_output(self, sample_probs: int, sample_idx: int) -> int:
        raise NotImplementedError()

    def test(self, test_probs: np.ndarray) -> np.ndarray:
        num_samples, num_outputs, _ = test_probs.shape
        assert num_outputs == self.num_outputs, 'Expected {} outputs. Got {}'.format(self.num_outputs, num_outputs)

        predictions: List[int] = []
        output_levels: List[int] = []

        for sample_idx, sample_probs in enumerate(test_probs):
            level = self.select_output(sample_probs=sample_probs,
                                       sample_idx=sample_idx)
            pred = np.argmax(sample_probs[level])

            predictions.append(pred)
            output_levels.append(level)

        output_counts = np.bincount(output_levels, minlength=self.num_outputs)
        observed_rates = output_counts / num_samples

        return EarlyExitResult(predictions=np.vstack(predictions).reshape(-1),
                               output_levels=np.vstack(output_levels).reshape(-1),
                               observed_rates=observed_rates)

    def fit_attack_model(self, val_outputs: np.ndarray, val_labels: np.ndarray, window_size: int, num_samples: int):
        inputs, labels = make_attack_dataset(outputs=val_outputs,
                                             labels=val_labels,
                                             window_size=window_size,
                                             num_samples=num_samples,
                                             rand=self._val_rand)

        self._attack_model.fit(inputs, labels)

    def test_attack_model(self, test_outputs: np.ndarray, test_labels: np.ndarray, window_size: int, num_samples: int) -> float:
        inputs, labels = make_attack_dataset(outputs=test_outputs,
                                             labels=test_labels,
                                             window_size=window_size,
                                             num_samples=num_samples,
                                             rand=self._test_rand)

        return self._attack_model.score(inputs, labels)


class RandomExit(EarlyExiter):

    def __init__(self, rates: List[float]):
        super().__init__(rates=rates)
        self._rand = np.random.RandomState(23471)
        self._output_idx = list(range(len(rates)))

    def select_output(self, sample_probs: np.ndarray, sample_idx: int) -> int:
        level = self._rand.choice(self._output_idx, size=1, p=self.rates)
        return int(level)


class ThresholdExiter(EarlyExiter):

    def __init__(self, rates: List[float]):
        super().__init__(rates=rates)
        self._thresholds = [0.0 for _ in rates]

    @property
    def thresholds(self) -> List[float]:
        return self._thresholds

    def set_threshold(self, t: float, level: int):
        assert level >= 0 and level < self.num_outputs, 'Level must be in [0, {})'.format(self.num_outputs)
        self._thresholds[level] = t


class EntropyExit(ThresholdExiter):

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        assert val_probs.shape[0] == val_labels.shape[0], 'Misaligned probabilites and labels'
        assert self.num_outputs == 2, 'Entropy Exiting only works with 2 outputs'
        assert val_probs.shape[1] == self.num_outputs, 'Expected {} outputs. Got {}'.format(self.num_outputs, val_probs.shape[1])

        # Compute the entropy for each predicted distribution
        pred_entropy = compute_entropy(val_probs, axis=-1)  # [B, L]

        # Comute the thresholds based on the quantile
        # TODO: Make this 'cumulative' by removing the samples stopped at the previous level(s)
        for level in range(self.num_outputs):
            t = np.quantile(pred_entropy[:, level], q=self.rates[level])
            self.set_threshold(t, level)

        # Catch everything at the last level
        self.set_threshold(BIG_NUMBER, self.num_outputs - 1)

    def select_output(self, sample_probs: np.ndarray, sample_idx: int) -> int:
        level_entropy = compute_entropy(sample_probs, axis=-1)  # [L]
        comp = np.less(level_entropy, self.thresholds).astype(int)  # [L]
        return np.argmax(comp)


class MaxProbExit(ThresholdExiter):

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        assert val_probs.shape[0] == val_labels.shape[0], 'Misaligned probabilites and labels'
        assert self.num_outputs == 2, 'Max Prob Exiting only works with 2 outputs'
        assert val_probs.shape[1] == self.num_outputs, 'Expected {} outputs. Got {}'.format(self.num_outputs, val_probs.shape[1])

        # Compute the maximum probability for each predicted distribution
        max_probs = np.max(val_probs, axis=-1)  # [B, L]

        # Comute the thresholds based on the quantile
        # TODO: Make this 'cumulative' by removing the samples stopped at the previous level(s)
        for level in range(self.num_outputs):
            t = np.quantile(max_probs[:, level], q=1.0 - self.rates[level])
            self.set_threshold(t, level)

        # Catch everything at the last level
        self.set_threshold(0.0, self.num_outputs - 1)

    def select_output(self, sample_probs: np.ndarray, sample_idx: int) -> int:
        level_probs = np.max(sample_probs, axis=-1)  # [L]
        comp = np.greater(level_probs, self.thresholds).astype(int)  # [L]
        return np.argmax(comp)


class LabelThresholdExiter(EarlyExiter):

    def __init__(self, rates: List[float]):
        super().__init__(rates=rates)
        self._thresholds: Optional[np.ndarray] = None

    @property
    def thresholds(self) -> np.ndarray:
        assert self._thresholds is not None, 'Must call init_thresholds() first'
        return self._thresholds

    def init_thresholds(self, num_labels: int):
        self._thresholds = np.zeros(shape=(self.num_outputs, num_labels))

    def set_threshold(self, t: float, level: int, label: int):
        assert self._thresholds is not None, 'Must call init_thresholds() first'
        self._thresholds[level, label] = t

    def get_threshold(self, level: int, label: int) -> float:
        assert self._thresholds is not None, 'Must call init_thresholds() first'
        return self._thresholds[level, label]

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_quantile(self, level: int) -> float:
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
            t = np.quantile(distribution, q=self.get_quantile(level=0))

            self.set_threshold(t=t, level=0, label=pred)
            self.set_threshold(t=0.0, level=1, label=pred)


class LabelMaxProbExit(LabelThresholdExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return np.max(probs, axis=-1)

    def get_quantile(self, level: int) -> int:
        return 1.0 - self.rates[level]

    def select_output(self, sample_probs: np.ndarray, sample_idx: int) -> int:
        # Get the maximum probabilities
        max_probs = np.max(sample_probs, axis=-1)  # [L]
        first_prob = float(max_probs[0])

        # Get the threshold
        first_pred = int(np.argmax(sample_probs[0, :]))
        t = self.get_threshold(level=0, label=first_pred)

        return int(first_prob < t)


class LabelEntropyExit(LabelThresholdExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_entropy(probs, axis=-1)

    def get_quantile(self, level: int) -> int:
        return self.rates[level]

    def select_output(self, sample_probs: np.ndarray, sample_idx: int) -> int:
        # Get the maximum probabilities
        entropies = compute_entropy(sample_probs, axis=-1)  # [L]
        first_entropy = float(entropies[0])

        # Get the threshold
        first_pred = int(np.argmax(sample_probs[0, :]))
        t = self.get_threshold(level=0, label=first_pred)

        return int(first_entropy > t)


def make_policy(strategy: ExitStrategy, rates: List[float]) -> EarlyExiter:
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
    else:
        raise ValueError('No policy {}'.format(strategy))
