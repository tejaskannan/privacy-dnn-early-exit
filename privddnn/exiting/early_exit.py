import numpy as np
import os.path
from collections import namedtuple, defaultdict, Counter
from enum import Enum, auto
from sklearn.ensemble import AdaBoostClassifier
from typing import List, Tuple, Dict, DefaultDict, Optional

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


EarlyExitResult = namedtuple('EarlyExitResult', ['predictions', 'output_levels', 'observed_rates'])


class ExitStrategy(Enum):
    RANDOM = auto()
    ENTROPY = auto()
    MAX_PROB = auto()
    LABEL_ENTROPY = auto()
    LABEL_MAX_PROB = auto()
    HYBRID_MAX_PROB = auto()
    HYBRID_ENTROPY = auto()
    GREEDY_EVEN = auto()
    EVEN_MAX_PROB = auto()
    EVEN_LABEL_MAX_PROB = auto()


class EarlyExiter:

    def __init__(self, rates: List[float]):
        assert np.isclose(np.sum(rates), 1.0), 'Rates must sum to 1'

        self._attack_model = AdaBoostClassifier()
        self._rates = rates
        self._prior = np.zeros(shape=(1, ))

    @property
    def rates(self) -> List[float]:
        return self._rates

    @property
    def num_outputs(self) -> int:
        return len(self._rates)

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        num_labels = val_probs.shape[-1]
        val_preds = np.argmax(val_probs[:, -1, :], axis=-1)  # [B]
        self._prior = np.bincount(val_preds, minlength=num_labels).astype(float)
        self._prior /= val_probs.shape[0]

    def select_output(self, probs: np.ndarray, rand_rate: float) -> int:
        raise NotImplementedError()

    def update(self, first_pred: int, pred: int, level: int):
        pass

    def reset(self, num_labels: int, pred_rates: np.ndarray):
        pass

    def get_level(self, probs: np.ndarray) -> Tuple[int, int]:
        assert len(probs.shape) == 1, 'Must provide a 1d array of predicted probabilities'
        reshaped_probs = probs.reshape(1, -1)  # [1, L]
        return self.select_output(probs=reshaped_probs, rand_rate=0.0)

    def get_prediction(self, probs: np.ndarray, level: int) -> int:
        return np.argmax(probs[level])

    def test(self, test_probs: np.ndarray, pred_rates: np.ndarray, max_num_samples: Optional[int]) -> np.ndarray:
        num_samples, num_outputs, num_labels = test_probs.shape
        assert num_outputs == self.num_outputs, 'Expected {} outputs. Got {}'.format(self.num_outputs, num_outputs)
        assert pred_rates.shape == (num_outputs, num_labels), 'Exit rates should be a ({}, {}) array. Got {}'.format(num_outputs, num_labels, pred_rates.shape)

        predictions: List[int] = []
        output_levels: List[int] = []

        self.reset(num_labels=num_labels, pred_rates=pred_rates)

        # Assume a uniform prior
        #controller = RandomnessController(targets=target_exit_rates[0],
        #                                  num_labels=num_labels,
        #                                  significance=self._significance,
        #                                  window=self._window)
        #controller.reset()

        for sample_idx, sample_probs in enumerate(test_probs):
            if (max_num_samples is not None) and (sample_idx >= max_num_samples):
                break

            # Get the randomness rate for this sample
            #rand_rate = controller.get_rate(sample_idx=sample_idx)
            rand_rate = 0.0

            level = self.select_output(probs=sample_probs, rand_rate=rand_rate)
            pred = self.get_prediction(probs=sample_probs, level=level)
            first_pred = np.argmax(sample_probs[0])

            self.update(first_pred=first_pred, pred=pred, level=level)

            # Update the controller if the level is 1
            #controller.update(pred=pred, level=level)

            predictions.append(pred)
            output_levels.append(level)

        output_counts = np.bincount(output_levels, minlength=self.num_outputs)
        observed_rates = output_counts / num_samples

        return EarlyExitResult(predictions=np.vstack(predictions).reshape(-1),
                               output_levels=np.vstack(output_levels).reshape(-1),
                               observed_rates=observed_rates)


class RandomExit(EarlyExiter):

    def __init__(self, rates: List[float]):
        super().__init__(rates=rates)
        self._output_idx = list(range(len(rates)))

    def select_output(self, probs: np.ndarray, rand_rate: float) -> int:
        level = np.random.choice(self._output_idx, size=1, p=self.rates)
        return int(level)


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

    def select_output(self, probs: np.ndarray, rand_rate: float) -> int:
        first_pred = np.argmax(probs[0])
        stay_rate = self._stay_counter[first_pred] / (self._stay_counter[first_pred] + self._elevate_counter[first_pred] + SMALL_NUMBER)

        return int(stay_rate > self.rates[0])


class EvenRandomizedExit(EarlyExiter):

    def __init__(self, rates: List[float]):
        super().__init__(rates=rates)
        self._output_idx = list(range(len(rates)))

    def select_output(self, probs: np.ndarray, rand_rate: float) -> int:
        level = np.random.choice(self._output_idx, size=1, p=self.rates)
        return int(level)

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        super().fit(val_probs=val_probs, val_labels=val_labels)

        assert val_probs.shape[0] == val_labels.shape[0], 'misaligned probabilites and labels'
        assert self.num_outputs == 2, 'threshold exiting only works with 2 outputs'
        assert val_probs.shape[1] == self.num_outputs, 'expected {} outputs. got {}'.format(self.num_outputs, val_probs.shape[1])

        rand_rates = fit_even_randomization(preds=np.argmax(val_probs, axis=-1), target=self.rates[1], num_labels=val_probs.shape[-1])


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

    def select_output(self, probs: np.ndarray, rand_rate: float) -> int:
        metric = self.compute_metric(probs)
        comp = np.greater(metric, self.thresholds).astype(int)  # [L]
        return np.argmax(comp)  # Breaks ties by selecting the first value

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


class EvenThresholdExiter(ThresholdExiter):

    def __init__(self, epsilon: float, horizon: int, rates: List[float]):
        super().__init__(rates=rates)
        self._epsilon = epsilon
        self._horizon = horizon
        self._stay_counter: Counter = Counter()
        self._elevate_counter: Counter = Counter()

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

        #print('Level: {}, Pred: {}, Stay: {}, Elevate: {}'.format(level, pred, self._stay_counter[pred], self._elevate_counter[pred]))

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
            #shift = self.horizon / 2.0
            width = self.horizon * 2.0

            self._prob_adjustments[0, pred] = linear_step(x=stay_diff, width=width, clip=1.0)
            self._prob_adjustments[1, pred] = linear_step(x=elevate_diff, width=width, clip=1.0)

    def get_prediction(self, probs: np.ndarray, level: int) -> int:
        level_probs = probs[level]  # [K]
        adjusted_probs = level_probs + mask_non_max(self._prob_adjustments[level])  # [K]

        return np.argmax(adjusted_probs)

        #if pred in self._preds_to_avoid:
        #    return int(np.random.choice(self._pred_space, p=self._pred_weights, size=1))
        #return pred

    def reset(self, num_labels: int, pred_rates: np.ndarray):
        self._stay_counter = Counter()
        self._elevate_counter = Counter()
        self._prior = np.eye(num_labels)
        self._prob_adjustments = np.zeros(shape=(2, num_labels))
        self._num_labels = num_labels

        #self._pred_rates = pred_rates
        #self._preds_to_avoid: Set[int] = set()

        #for pred in range(num_labels):
        #    first_rate = pred_rates[0, pred]
        #    second_rate = pred_rates[1, pred]

        #    max_stop_rate = (first_rate) / max(first_rate + (1.0 - self.rates[0]) * second_rate, SMALL_NUMBER)

        #    if (max_stop_rate < (self.rates[0] - self.epsilon)) and (self.rates[0] < (1.0 - SMALL_NUMBER)) and (self.rates[0] > SMALL_NUMBER):
        #        self._preds_to_avoid.add(pred)

        #self._pred_space = list(range(num_labels))
        #self._pred_weights = np.array([float(pred not in self._preds_to_avoid) for pred in range(num_labels)])
        #self._pred_weights /= np.sum(self._pred_weights)

    def select_output(self, probs: np.ndarray, rand_rate: float):
        adjusted_probs = mask_non_max(self._prob_adjustments) + probs

        first_pred = np.argmax(adjusted_probs[0])
        total_count = self._stay_counter[first_pred] + self._elevate_counter[first_pred] + 1
        stay_rate = (self._stay_counter[first_pred] + 1) / total_count
        stay_cost = abs(stay_rate - self.rates[0])

        # Compute the expected cost of elevating
        #elevate_probs = self._prior[first_pred] / np.sum(self._prior[first_pred])  # [L]
        #expected_rate = 0.0

        #for pred in range(len(elevate_probs)):
        #    expected_rate += elevate_probs[pred] * ((self._elevate_counter[pred]) / (self._stay_counter[pred] + self._elevate_counter[pred] + SMALL_NUMBER))

        #elevate_cost = abs(expected_rate - self.rates[1])

        #total_count = self._stay_counter[first_pred] + self._elevate_counter[first_pred]

        # Compute the level from both data-dependent exiting and even-ness exiting
        # to avoid timing attacks against this conditional behavior
        policy_level = super().select_output(probs=probs, rand_rate=rand_rate)
        even_level = int(stay_rate > self.rates[0])
        #even_level = int(elevate_cost < stay_cost)

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

    def select_output(self, probs: np.ndarray, rand_rate: float) -> int:
        # Get the threshold
        first_probs = probs[0, :]
        first_pred = int(np.argmax(first_probs))
        t = self.get_threshold(level=0, label=first_pred)

        # Get the metric on the first prediction
        metric = self.compute_metric(first_probs)

        return int(metric < t)


class LabelMaxProbExit(LabelThresholdExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_max_prob_metric(probs=probs)


class LabelEntropyExit(LabelThresholdExiter):

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        return compute_entropy_metric(probs=probs)


class EvenLabelThresholdExiter(LabelThresholdExiter):

    def __init__(self, epsilon: float, rates: List[float]):
        super().__init__(rates=rates)
        self._epsilon = epsilon
        self._stay_counter: Counter = Counter()
        self._elevate_counter: Counter = Counter()

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def update(self, first_pred: int, pred: int, level: int):
        if level == 0:
            self._stay_counter[pred] += 1
        else:
            self._elevate_counter[pred] += 1
            self._prior[first_pred, pred] += 1

    def get_prediction(self, preds: np.ndarray, level: int) -> int:
        pred = preds[level]

        if pred in self._preds_to_avoid:
            return int(np.random.choice(self._pred_space, p=self._pred_weights, size=1))
        return pred

    def reset(self, num_labels: int, pred_rates: np.ndarray):
        self._stay_counter = Counter()
        self._elevate_counter = Counter()
        self._prior = np.eye(num_labels)

        self._pred_rates = pred_rates
        self._preds_to_avoid: Set[int] = set()

        for pred in range(num_labels):
            first_rate = pred_rates[0, pred]
            second_rate = pred_rates[1, pred]

            max_stop_rate = (first_rate) / max(first_rate + (1.0 - self.rates[0]) * second_rate, SMALL_NUMBER)

            if (max_stop_rate < (self.rates[0] - self.epsilon)) and (self.rates[0] < (1.0 - SMALL_NUMBER)) and (self.rates[0] > SMALL_NUMBER):
                self._preds_to_avoid.add(pred)

        self._pred_space = list(range(num_labels))
        self._pred_weights = np.array([float(pred not in self._preds_to_avoid) for pred in range(num_labels)])
        self._pred_weights /= np.sum(self._pred_weights)

    def select_output(self, probs: np.ndarray, rand_rate: float):
        first_pred = np.argmax(probs[0])
        stay_rate = (self._stay_counter[first_pred]) / (self._stay_counter[first_pred] + self._elevate_counter[first_pred] + SMALL_NUMBER)
        stay_cost = abs(stay_rate - self.rates[0])

        # Compute the expected cost of elevating
        elevate_probs = self._prior[first_pred] / np.sum(self._prior[first_pred])  # [L]
        expected_rate = 0.0

        for pred in range(len(elevate_probs)):
            expected_rate += elevate_probs[pred] * ((self._elevate_counter[pred]) / (self._stay_counter[pred] + self._elevate_counter[pred] + SMALL_NUMBER))

        elevate_cost = abs(expected_rate - self.rates[1])

        # Compute the level from both data-dependent exiting and even-ness exiting
        # to avoid timing attacks against this conditional behavior
        policy_level = super().select_output(probs=probs, rand_rate=rand_rate)
        even_level = int(stay_rate > self.rates[0])

        if (stay_cost < self.epsilon) and (elevate_cost < self.epsilon):
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

    def select_output(self, probs: np.ndarray, rand_rate: float) -> int:
        if self._rand_rate is None:
            return super().select_output(probs, rand_rate=rand_rate)

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
        return EvenMaxProbExit(rates=rates, epsilon=0.001, horizon=5)
    elif strategy == ExitStrategy.EVEN_LABEL_MAX_PROB:
        return EvenLabelMaxProbExit(rates=rates, epsilon=0.001)
    else:
        raise ValueError('No policy {}'.format(strategy))
