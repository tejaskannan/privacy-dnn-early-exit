import numpy as np
import os.path
from collections import namedtuple, defaultdict
from enum import Enum, auto
from sklearn.ensemble import AdaBoostClassifier
from typing import List, Tuple, Dict, DefaultDict, Optional

from privddnn.utils.metrics import compute_entropy, create_confusion_matrix, sigmoid
from privddnn.utils.metrics import compute_max_prob_metric, compute_entropy_metric
from privddnn.utils.constants import BIG_NUMBER, SMALL_NUMBER
from privddnn.utils.file_utils import read_json
from privddnn.controllers.runtime_controller import RandomnessController
from .even_optimizer import fit_thresholds, fit_threshold_randomization
from .prob_optimizer import fit_prob_thresholds
from .threshold_optimizer import fit_thresholds_grad, fit_randomization, BETA


EarlyExitResult = namedtuple('EarlyExitResult', ['predictions', 'output_levels', 'observed_rates'])


class ExitStrategy(Enum):
    RANDOM = auto()
    ENTROPY = auto()
    MAX_PROB = auto()
    LABEL_ENTROPY = auto()
    LABEL_MAX_PROB = auto()
    HYBRID_MAX_PROB = auto()
    HYBRID_ENTROPY = auto()


class EarlyExiter:

    def __init__(self, rates: List[float]):
        assert np.isclose(np.sum(rates), 1.0), 'Rates must sum to 1'

        self._attack_model = AdaBoostClassifier()
        self._val_rand = np.random.RandomState(seed=28116)
        self._test_rand = np.random.RandomState(seed=52190)
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

    def select_output(self, probs: int, rand_rate: float) -> int:
        raise NotImplementedError()

    def test(self, test_probs: np.ndarray) -> np.ndarray:
        num_samples, num_outputs, num_labels = test_probs.shape
        assert num_outputs == self.num_outputs, 'Expected {} outputs. Got {}'.format(self.num_outputs, num_outputs)

        predictions: List[int] = []
        output_levels: List[int] = []

        # Assume a uniform prior
        #label_prior = np.ones(shape=(num_labels, )) / num_labels
        #controller = RandomnessController(prior=self._prior,
        #                                  target=self.rates[1],
        #                                  window=100,
        #                                  epsilon=0.002)
        #controller.reset()

        for sample_idx, sample_probs in enumerate(test_probs):
            # Get the randomness rate for this sample
            #rand_rate = controller.get_rate(sample_idx=sample_idx)

            level = self.select_output(probs=sample_probs, rand_rate=0.0)
            pred = np.argmax(sample_probs[level])

            # Update the controller if the level is 1
            #if level == 1:
            #    controller.update(pred=pred)

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
        self._rand = np.random.RandomState(23471)
        self._output_idx = list(range(len(rates)))

    def select_output(self, probs: np.ndarray, rand_rate: float) -> int:
        level = self._rand.choice(self._output_idx, size=1, p=self.rates)
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

    def select_output(self, probs: np.ndarray, rand_rate: float) -> int:
        metric = self.compute_metric(probs)
        comp = np.greater(metric, self.thresholds).astype(int)  # [L]
        return np.argmax(comp)  # Breaks ties by selecting the first value

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray):
        super().fit(val_probs=val_probs, val_labels=val_labels)

        assert val_probs.shape[0] == val_labels.shape[0], 'Misaligned probabilites and labels'
        assert self.num_outputs == 2, 'Threshold Exiting only works with 2 outputs'
        assert val_probs.shape[1] == self.num_outputs, 'Expected {} outputs. Got {}'.format(self.num_outputs, val_probs.shape[1])

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


class HybridRandomExit(LabelThresholdExiter):

    def __init__(self, rates: List[float], path: Optional[str], metric_name: str):
        super().__init__(rates=rates)
        assert metric_name in ('max-prob', 'entropy'), 'Metric name must be `max-prob` or `entropy`'

        self._thresholds: Optional[np.ndarray] = None
        self._rand_rate: Optional[np.ndarray] = None
        self._observed_rates: Optional[np.ndarray] = None
        self._trials = 1
        self._metric_name = metric_name

        self._rand = np.random.RandomState(seed=2890)
        self._noise_scale = 0.02
        self._epsilon = 0.002

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
        level = int(self._rand.uniform() < level_prob)

        # Determine whether we should act randomly
        should_act_random = int(self._rand.uniform() < self._rand_rate)
        rand_level = int(self._rand.uniform() < self.rates[1])

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

        for lr in learning_rates:
            for trial in range(self._trials):
                start_thresholds = np.copy(self.thresholds[0])

                if trial > 0:
                    start_thresholds += self._rand.uniform(low=-1 * self._noise_scale,
                                                           high=self._noise_scale,
                                                           size=start_thresholds.shape)

                loss, thresholds, weights, rates = fit_thresholds_grad(probs=val_probs[:, 0, :],
                                                              labels=val_labels,
                                                              target=target,
                                                              start_thresholds=start_thresholds,
                                                              learning_rate=lr,
                                                              metric_name=self.metric_name)

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

        rand_rate = fit_randomization(avg_rates=best_rates,
                                      labels=val_labels,
                                      epsilon=self._epsilon,
                                      target=target)

        print('Randomness Rate: {}'.format(rand_rate))
        print('Weights: {}'.format(np.square(best_weights)))

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
    else:
        raise ValueError('No policy {}'.format(strategy))
