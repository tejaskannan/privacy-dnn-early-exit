import numpy as np
from collections import namedtuple, defaultdict, Counter
from typing import Dict, DefaultDict, List, Tuple

from privddnn.utils.constants import SMALL_NUMBER
from privddnn.utils.metrics import softmax


MAX_ITERS = 50
PATIENCE = 15
NEIGHBORHOOD = 15
PRECISION = 64
TEMP_PRECISION = 64

EvalResult = namedtuple('EvalResult', ['num_correct', 'num_samples', 'level_counts', 'label_counts'])


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


def eval_on_split(data_split: DataSplit, threshold: float) -> EvalResult:
    """
    Executes a Maximum Probability stopping criteria using the given threshold.
    """
    max_first_probs = np.max(data_split.probs[:, 0, :], axis=-1)  # [B]
    num_levels = np.less(max_first_probs, threshold).astype(float)  # [B]

    level_preds = np.argmax(data_split.probs, axis=-1)  # [B, L]
    preds = (1.0 - num_levels) * level_preds[:, 0] + num_levels * level_preds[:, 1]  # [B]
    preds = preds.astype(int)  # [B]

    num_correct = np.sum(np.isclose(preds, data_split.labels))
    num_samples = preds.shape[0]

    num_labels = data_split.probs.shape[-1]
    level_counts = np.zeros(shape=(num_labels, ))
    label_counts = np.zeros(shape=(num_labels, ))

    for label, num_levels in zip(data_split.labels, num_levels):
        level_counts[label] += num_levels
        label_counts[label] += 1

    return EvalResult(num_correct=num_correct, num_samples=num_samples, level_counts=level_counts, label_counts=label_counts)


def get_stop_rates(pred_evals: Dict[int, EvalResult], num_labels: int) -> np.ndarray:
    level_counts = np.zeros(shape=(num_labels, ))
    label_counts = np.zeros(shape=(num_labels, ))

    for eval_result in pred_evals.values():
        level_counts += eval_result.level_counts
        label_counts += eval_result.label_counts

    return level_counts / (label_counts + SMALL_NUMBER)


def loss_fn(pred_evals: Dict[int, EvalResult], num_labels: int, target: float, should_print: bool = False) -> Tuple[float, int]:
    avg_level = get_stop_rates(pred_evals, num_labels=num_labels)

    #if should_print:
    #    print(avg_level)

    diff = np.abs(avg_level - target)
    mean_diff = np.abs(avg_level - np.average(avg_level))
    loss = diff + mean_diff

    return np.sum(loss), np.argmax(loss), avg_level


def fit_temperature(probs: np.ndarray, labels: np.ndarray, rate: float, start_temp: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fits temperature values to get all labels to stop at the first level with the given rate.
    """
    assert len(start_temp.shape) == 1, 'Must provide a 1d temperature array'
    assert len(thresholds.shape) == 1, 'Must provide a 1d thresholds array'
    num_labels = probs.shape[-1]
    target = 1.0 - rate

    rand = np.random.RandomState(89109)

    # Split the dataset by prediction
    splits = split_by_prediction(probs=probs, labels=labels)

    # Evaluate the initial temperature
    temperature = np.copy(start_temp)

    pred_evals: Dict[int, EvalResult] = dict()
    for pred, data_split in splits.items():
        data_split.apply_temperature(temperature[pred])
        pred_evals[pred] = eval_on_split(data_split, threshold=thresholds[pred])
        data_split.revert()

    prev_loss, worst_pred = loss_fn(pred_evals, num_labels, target=target, should_print=True)
    pred = worst_pred
    final_loss = prev_loss

    num_not_improved = 0
    neighborhood = NEIGHBORHOOD

    for i in range(MAX_ITERS):
        best_temp = temperature[pred]
        best_loss = prev_loss

        for n in range(-1 * neighborhood, neighborhood):
            temp = temperature[pred] + (n / TEMP_PRECISION)

            splits[pred].apply_temperature(temp)
            eval_result = eval_on_split(splits[pred], threshold=thresholds[pred])
            splits[pred].revert()

            pred_evals[pred] = eval_result
            loss, _ = loss_fn(pred_evals, num_labels=num_labels, target=target)

            if loss < best_loss:
                best_temp = temp
                best_loss = loss

        temperature[pred] = best_temp

        print('\nBEST TEMP: {}, Pred: {}'.format(best_temp, pred))

        splits[pred].apply_temperature(temp)
        best_eval = eval_on_split(splits[pred], threshold=thresholds[pred])
        splits[pred].revert()

        pred_evals[pred] = best_eval

        final_loss, worst_pred = loss_fn(pred_evals, num_labels=num_labels, target=target, should_print=True)

        print('Loss: {}'.format(prev_loss), end='\r')

        if abs(final_loss - prev_loss) < 1e-5:
            num_not_improved += 1
            pred = rand.randint(0, num_labels)
        else:
            num_not_improved = 0
            pred = worst_pred

        prev_loss = final_loss
        neighborhood = max(neighborhood - 1, NEIGHBORHOOD)

        if num_not_improved >= PATIENCE:
            print('\nConverged')
            break

    return temperature, final_loss

def fit_thresholds(probs: np.ndarray, labels: np.ndarray, rate: float, start_thresholds: np.ndarray, temperature: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fits thresholds to get all labels to stop at the first level with the given rate.
    """
    assert len(start_thresholds.shape) == 1, 'Must provide a 1d thresholds array'
    assert len(temperature.shape) == 1, 'Must provide a 1d temperature array'
    num_labels = probs.shape[-1]
    target = 1.0 - rate

    # Split the dataset by prediction
    splits = split_by_prediction(probs=probs, labels=labels)

    # Scale all the probabilities by the temperature (this doesn't change later on)
    for pred, data_split in splits.items():
        data_split.apply_temperature(temperature[pred])

    # Copy the starting thresholds (so we can make non-destructive changes)
    thresholds = np.copy(start_thresholds)
    rand = np.random.RandomState(seed=148)

    # Evaluate the initial thresholds
    pred_evals: Dict[int, EvalResult] = dict()
    for pred, data_split in splits.items():
        pred_evals[pred] = eval_on_split(data_split, threshold=thresholds[pred])

    prev_loss, worst_pred, obs_rates = loss_fn(pred_evals, num_labels, target=target, should_print=True)
    pred = worst_pred
    final_loss = prev_loss

    num_not_improved = 0
    neighborhood = NEIGHBORHOOD

    for i in range(MAX_ITERS):
        best_t = thresholds[pred]
        best_loss = prev_loss

        for n in range(-1 * neighborhood, neighborhood):
            t = thresholds[pred] + (n / PRECISION)
            eval_result = eval_on_split(splits[pred], threshold=t)

            pred_evals[pred] = eval_result
            loss, _, _ = loss_fn(pred_evals, num_labels=num_labels, target=target)

            if loss < best_loss:
                best_t = t
                best_loss = loss

        thresholds[pred] = best_t
        best_eval = eval_on_split(splits[pred], threshold=best_t)
        pred_evals[pred] = best_eval

        final_loss, worst_pred, obs_rates = loss_fn(pred_evals, num_labels=num_labels, target=target, should_print=True)

        print('Loss: {}'.format(prev_loss), end='\r')

        if abs(final_loss - prev_loss) < 1e-5:
            num_not_improved += 1
            pred = rand.randint(0, num_labels)
        else:
            num_not_improved = 0
            pred = worst_pred

        prev_loss = final_loss
        neighborhood = max(neighborhood - 1, NEIGHBORHOOD)

        if num_not_improved >= PATIENCE:
            print('\nConverged')
            break

    return thresholds, final_loss, obs_rates
