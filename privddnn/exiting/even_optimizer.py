import numpy as np
from collections import namedtuple, defaultdict, Counter
from typing import Dict, DefaultDict, List, Tuple

from privddnn.utils.constants import SMALL_NUMBER


MAX_ITERS = 50
PATIENCE = 5
NEIGHBORHOOD = 10
PRECISION = 64

DataSplit = namedtuple('DataSplit', ['probs', 'labels'])
EvalResult = namedtuple('EvalResult', ['num_correct', 'num_samples', 'level_counts', 'label_counts'])


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


def loss_fn(pred_evals: Dict[int, EvalResult], num_labels: int, target: float) -> Tuple[float, int]:
    avg_level = get_stop_rates(pred_evals, num_labels=num_labels)
    diff = np.abs(avg_level - target)
    return np.sum(diff), np.argmax(diff)


def fit_thresholds(probs: np.ndarray, labels: np.ndarray, rate: float, start_thresholds: np.ndarray) -> np.ndarray:
    """
    Fits thresholds to get all labels to stop at the first level with the given rate.
    """
    assert len(start_thresholds.shape) == 1, 'Must provide a 1d thresholds array'
    num_labels = probs.shape[-1]
    target = 1.0 - rate

    # Split the dataset by prediction
    splits = split_by_prediction(probs=probs, labels=labels)

    prev_thresholds = np.copy(start_thresholds)
    thresholds = np.copy(start_thresholds)

    # Evaluate the initial thresholds
    pred_evals: Dict[int, EvalResult] = dict()
    for pred, data_split in splits.items():
        pred_evals[pred] = eval_on_split(data_split, threshold=thresholds[pred])

    rand = np.random.RandomState(seed=148)
    prev_loss, worst_pred = loss_fn(pred_evals, num_labels, target=target)
    num_not_improved = 0

    for _ in range(MAX_ITERS):
        pred = worst_pred
        best_t = thresholds[pred]
        best_loss = prev_loss

        for n in range(-1 * NEIGHBORHOOD, NEIGHBORHOOD):
            t = thresholds[pred] + (n / PRECISION)
            eval_result = eval_on_split(splits[pred], threshold=t)

            pred_evals[pred] = eval_result
            loss, _ = loss_fn(pred_evals, num_labels=num_labels, target=target)

            if loss < best_loss:
                best_t = t
                best_loss = loss

        thresholds[pred] = best_t
        best_eval = eval_on_split(splits[pred], threshold=best_t)
        pred_evals[pred] = best_eval

        prev_loss, worst_pred = loss_fn(pred_evals, num_labels=num_labels, target=target)

        print('Loss: {}'.format(prev_loss), end='\r')

        if np.sum(np.abs(thresholds, prev_thresholds)) < 1e-5:
            num_not_improved += 1
        else:
            num_not_improved = 0

        prev_thresholds = np.copy(thresholds)

        if num_not_improved >= PATIENCE:
            print('Converged')
            break

    return thresholds
