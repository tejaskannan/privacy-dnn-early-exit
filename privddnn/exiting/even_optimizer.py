import numpy as np
import scipy.stats as stats
import math
from collections import namedtuple, defaultdict, Counter
from typing import Dict, DefaultDict, List, Tuple

from privddnn.utils.constants import SMALL_NUMBER, BIG_NUMBER
from privddnn.utils.metrics import softmax, create_confusion_matrix


MAX_ITERS = 25
PATIENCE = 15
NEIGHBORHOOD = 25
PRECISION = 64
TEMP_PRECISION = 64

EvalResult = namedtuple('EvalResult', ['level_counts', 'label_counts'])


class DataSplit:

    def __init__(self, probs: np.ndarray, labels: np.ndarray):
        self.probs = probs
        self.labels = labels
        self.original_probs = np.copy(probs)
        self.logits = np.log(np.maximum(probs, 1e-5))

    def apply_temperature(self, temp: float):
        self.probs = softmax(self.logits - temp, axis=-1)

    def revert(self):
        self.probs = self.original_probs


def n_choose_k(n: int, k: int) -> int:
    return math.factorial(n) / (math.factorial(n - k) * math.factorial(k))


def split_by_prediction(probs: np.ndarray, labels: np.ndarray) -> Dict[int, DataSplit]:
    """
    Splits the given dataset by the first-level predictions.
    """
    probs_dict: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
    labels_dict: DefaultDict[int, List[int]] = defaultdict(list)

    splits: List[DataSplit] = []
    num_labels = probs.shape[-1]

    for sample_probs, label in zip(probs, labels):
        pred = np.argmax(sample_probs[0])
        probs_dict[pred].append(np.expand_dims(sample_probs, axis=0))
        labels_dict[pred].append(label)

    results: Dict[int, DataSplit] = dict()
    for pred in sorted(probs_dict.keys()):
        split = DataSplit(probs=np.vstack(probs_dict[pred]),
                          labels=np.vstack(labels_dict[pred]))
        results[pred] = split

    return results


def eval_on_split(data_split: DataSplit, thresholds: List[float], sample_probs: np.ndarray, rand: np.random.RandomState) -> EvalResult:
    """
    Executes a Maximum Probability stopping criteria using the given threshold.
    """
    max_first_probs = np.max(data_split.probs[:, 0, :], axis=-1)  # [B]

    num_labels = data_split.probs.shape[-1]
    level_counts = np.zeros(shape=(num_labels, ))
    label_counts = np.zeros(shape=(num_labels, ))

    pred = np.argmax(sample_probs)

    for max_prob, label in zip(max_first_probs, data_split.labels):
        threshold = rand.choice(thresholds, size=1, replace=False, p=sample_probs)

        num_levels = int(max_prob < threshold)

        level_counts[label] += num_levels
        label_counts[label] += 1

    return EvalResult(level_counts=level_counts, label_counts=label_counts)


def evaluate(max_probs: np.ndarray, preds: np.ndarray, labels: np.ndarray, thresholds: List[float], sample_probs: np.ndarray, rand: np.random.RandomState, num_labels: int) -> np.ndarray:
    """
    Executes a Maximum Probability stopping criteria using the given threshold.
    """
    assert max_probs.shape == labels.shape, 'Max Probs and Labels must have the same shape'
    assert preds.shape == labels.shape, 'Preds and Labels must have the same shape'

    level_counts = np.zeros(shape=(num_labels, ))
    label_counts = np.zeros(shape=(num_labels, ))

    for max_prob, pred, label in zip(max_probs, preds, labels):
        threshold = rand.choice(thresholds, size=1, replace=False, p=sample_probs[pred])
        num_levels = int(max_prob < threshold)

        level_counts[label] += num_levels
        label_counts[label] += 1

    return level_counts / (label_counts + SMALL_NUMBER)


def evaluate_rand(max_probs: np.ndarray,
                  preds: np.ndarray,
                  labels: np.ndarray,
                  thresholds: List[float],
                  sample_probs: np.ndarray,
                  rand_rate: float,
                  target: float,
                  rand: np.random.RandomState,
                  num_labels: int) -> np.ndarray:
    """
    Executes a Maximum Probability stopping criteria using the given threshold.
    """
    assert max_probs.shape == labels.shape, 'Max Probs and Labels must have the same shape'
    assert preds.shape == labels.shape, 'Preds and Labels must have the same shape'

    level_counts = np.zeros(shape=(num_labels, ))
    label_counts = np.zeros(shape=(num_labels, ))

    for max_prob, pred, label in zip(max_probs, preds, labels):
        r = rand.uniform()

        if r < rand_rate:
            r1 = rand.uniform()
            num_levels = int(r1 < target)
        else:
            threshold = rand.choice(thresholds, size=1, replace=False, p=sample_probs[pred])
            num_levels = int(max_prob < threshold)

        level_counts[label] += num_levels
        label_counts[label] += 1

    return level_counts / (label_counts + SMALL_NUMBER)


def eval_on_split_rand(data_split: DataSplit, thresholds: List[float], sample_probs: np.ndarray, rand_rate: float, target: float, rand: np.random.RandomState) -> EvalResult:
    """
    Executes a Maximum Probability stopping criteria using the given threshold.
    """
    max_first_probs = np.max(data_split.probs[:, 0, :], axis=-1)  # [B]

    num_labels = data_split.probs.shape[-1]
    level_counts = np.zeros(shape=(num_labels, ))
    label_counts = np.zeros(shape=(num_labels, ))

    for max_prob, label in zip(max_first_probs, data_split.labels):
        r = rand.uniform()

        if r < rand_rate:
            r2 = rand.uniform()
            num_levels = int(r2 < target)
        else:
            threshold = rand.choice(thresholds, size=1, replace=False, p=sample_probs)
            num_levels = int(max_prob < threshold)

        level_counts[label] += num_levels
        label_counts[label] += 1

    return EvalResult(level_counts=level_counts, label_counts=label_counts)


def get_stop_rates(pred_evals: Dict[int, EvalResult], num_labels: int) -> np.ndarray:
    level_counts = np.zeros(shape=(num_labels, ))
    label_counts = np.zeros(shape=(num_labels, ))

    for eval_result in pred_evals.values():
        level_counts += eval_result.level_counts
        label_counts += eval_result.label_counts

    return level_counts / (label_counts + SMALL_NUMBER)


def loss_fn(avg_level: np.ndarray, num_labels: int, target: float, should_print: bool = False) -> Tuple[float, int]:
    diff = np.abs(avg_level - target)

    max_diff = np.zeros_like(diff)
    for i in range(len(max_diff)):
        max_diff[i] = np.max(np.abs(avg_level - avg_level[i]))

    loss = diff + max_diff

    return np.sum(loss), np.argmax(loss)


def most_freq_pred(probs: np.ndarray, labels: np.ndarray) -> List[int]:
    first_probs = probs[:, 0, :]
    num_labels = first_probs.shape[-1]
    preds = np.argmax(first_probs, axis=-1)
    correct = (preds == labels).astype(int)

    result: List[int] = []

    for label in range(num_labels):
        mask = np.equal(labels, label)
        masked_preds = np.where(mask, preds, num_labels)
        pred_counts = np.bincount(masked_preds, minlength=num_labels + 1)
        most_freq_pred = np.argmax(pred_counts[0:-1])

        #mask = (labels == label).astype(int)
        #accuracy = np.sum(correct * mask) / np.sum(mask)

        result.append(most_freq_pred)

    return result


def fit_thresholds(probs: np.ndarray, labels: np.ndarray, rate: float, start_thresholds: np.ndarray, temperature: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fits thresholds to get all labels to stop at the first level with the given rate.
    """
    assert len(start_thresholds.shape) == 1, 'Must provide a 1d thresholds array'
    num_labels = probs.shape[-1]
    target = 1.0 - rate

    # Get the prediction confusion matrix for the first level
    max_probs = np.max(probs[:, 0, :], axis=-1)
    preds = np.argmax(probs[:, 0, :], axis=-1)
    confusion_mat = create_confusion_matrix(predictions=preds, labels=labels)
    confusion_mat /= (np.sum(confusion_mat, axis=-1, keepdims=True) + SMALL_NUMBER)

    # Copy the starting thresholds (so we can make non-destructive changes)
    thresholds = np.copy(start_thresholds)
    rand = np.random.RandomState(seed=148)

    # Evaluate the initial thresholds
    eval_result = evaluate(max_probs=max_probs, preds=preds, labels=labels, sample_probs=confusion_mat, thresholds=thresholds, rand=rand, num_labels=num_labels)

    prev_loss, worst_pred = loss_fn(eval_result, num_labels, target=target, should_print=True)
    final_loss = prev_loss
    pred = rand.randint(low=0, high=num_labels)
    best_rates = eval_result

    num_not_improved = 0
    neighborhood = 15
    sample_idx = np.arange(len(labels))

    for i in range(MAX_ITERS):
        best_t = thresholds[pred]
        best_loss = prev_loss

        sample_probs = confusion_mat[pred]

        # Evaluate this threshold
        upper = 1.0
        lower = 0.0
        t = best_t

        for _ in range(neighborhood):
            thresholds[pred] = t

            rates: List[np.ndarray] = []

            for _ in range(4):
                batch_idx = rand.choice(sample_idx, size=1024, replace=False)
                batch_max_probs = max_probs[batch_idx]
                batch_preds = preds[batch_idx]
                batch_labels = labels[batch_idx]

                eval_result = evaluate(max_probs=batch_max_probs,
                                       preds=batch_preds,
                                       labels=batch_labels,
                                       sample_probs=confusion_mat,
                                       thresholds=thresholds,
                                       rand=rand,
                                       num_labels=num_labels)

                rates.append(np.expand_dims(eval_result, axis=0))

            avg_rates = np.average(np.vstack(rates), axis=0)

            #print('Avg Rates: {}'.format(avg_rates))
            #print('Thresholds: {}'.format(thresholds))
            #print('==========')

            loss, _ = loss_fn(avg_rates, num_labels=num_labels, target=target)

            if loss < best_loss:
                best_t = t
                best_loss = loss
                best_rates = avg_rates

            if avg_rates[pred] > target:
                upper = t
            else:
                lower = t

            if abs(upper - lower) < 1e-3:
                break

            t = (upper + lower) / 2.0

        print('Iteration: {}, Loss: {:.6f}'.format(i, best_loss), end='\n')
        #print('Rates: {}'.format(best_rates))

        if abs(best_loss - prev_loss) < 1e-5:
            num_not_improved += 1
        else:
            num_not_improved = 0

        prev_loss = best_loss
        neighborhood = min(neighborhood + 1, NEIGHBORHOOD)
        pred = rand.randint(0, num_labels)

        if num_not_improved >= PATIENCE:
            print('\nConverged')
            break

    return thresholds, best_loss, best_rates


def fit_threshold_randomization(target: float, elevation_rates: np.ndarray, epsilon: float) -> float:
    """
    Fits thresholds to get all labels to stop at the first level with the given rate.
    """
    assert len(elevation_rates.shape) == 1, 'Must provide a 1d array of elevation rates'

    abs_diffs = np.abs(elevation_rates - target)
    worst_idx = np.argmax(abs_diffs)
    label_rate = elevation_rates[worst_idx]

    optimal_rate = 1.0 - abs(epsilon / (label_rate - target + SMALL_NUMBER))
    optimal_rate = max(min(optimal_rate, 1.0), 0.0)  # Clip into the range [0, 1]

    return optimal_rate
