import numpy as np
from collections import namedtuple, defaultdict
from typing import List, Tuple, Dict, DefaultDict

from privddnn.utils.metrics import compute_entropy
from privddnn.utils.constants import BIG_NUMBER


EarlyExitResult = namedtuple('EarlyExitResult', ['preds', 'output_counts'])


def random_exit(test_probs: np.ndarray, rates: List[float], rand: np.random.RandomState) -> EarlyExitResult:
    """
    Performs early-exiting random of the input label.

    Args:
        test_probs: A [B, L, K] array of predicted probabilities on the test set
        rates: A [L] list of stopping rates (must sum to 1)
        rand: The random state used for determining stopping with reproducible results
    Returns:
        A [B] array of predictions based on the stopping criteria and a [B] array
        of the output indices for each prediction
    """
    num_samples, num_outputs, num_classes = test_probs.shape

    assert len(rates) == num_outputs, 'Must provide {} rates. Got: {}'.format(len(rates), num_outputs)
    assert np.isclose(np.sum(rates), 1.0), 'Rates must sum to 1'

    predictions: List[int] = []
    output_counts: List[int] = []

    for sample_probs in test_probs:
        r = rand.uniform()
        output_idx = 0

        rate_sum = 0.0
        for idx in range(num_outputs):
            rate_sum += rates[idx]

            if r <= rate_sum:
                break

            output_idx += 1

        output_idx = min(output_idx, num_outputs - 1)
        pred = np.argmax(sample_probs[output_idx])

        predictions.append(pred)
        output_counts.append(output_idx)

    return EarlyExitResult(preds=np.vstack(predictions).reshape(-1),
                           output_counts=np.vstack(output_counts).reshape(-1))


def entropy_exit(val_probs: np.ndarray, test_probs: np.ndarray, rates: List[float]) -> EarlyExitResult:
    """
    Performs early existing using thresholds on entropy of the predicted probs.

    Args:
        val_probs: A [C, L, K] array of predicted probabilities on the validation set
        test_probs: A [B, L, K] array of predicted probabilities on the test set
        rates: A [L] array of stopping rates (must sum to 1)
    Returns:
        A [B] array of predictions and a [B] array of the output indices for each
        prediction.
    """
    num_samples, num_outputs, num_classes = test_probs.shape

    assert len(rates) == num_outputs, 'Must provide {} rates. Got: {}'.format(len(rates), num_outputs)
    assert np.isclose(np.sum(rates), 1.0), 'Rates must sum to 1'
    assert num_outputs == 2, 'Only supports 2 outputs'

    # Compute the thresholds. We use the validation set which is available offline
    val_entropy = compute_entropy(val_probs, axis=-1)  # [B, L]

    t0 = np.quantile(val_entropy[:, 0], q=rates[0])
    thresholds = [t0, BIG_NUMBER]

    # Get the predictions
    test_entropy = compute_entropy(test_probs, axis=-1)
    test_preds = np.argmax(test_probs, axis=-1)

    predictions: List[int] = []
    output_counts: List[int] = []

    for sample_idx in range(num_samples):
        sample_entropy = test_entropy[sample_idx]  # [L]
        stop_comparison = np.less(sample_entropy, thresholds)
        stopped_idx = np.argmax(stop_comparison)
        pred = test_preds[sample_idx, stopped_idx]

        predictions.append(pred)
        output_counts.append(stopped_idx)

    return EarlyExitResult(preds=np.vstack(predictions).reshape(-1),
                           output_counts=np.vstack(output_counts).reshape(-1))


def max_prob_exit(val_probs: np.ndarray, test_probs: np.ndarray, rates: List[float]) -> EarlyExitResult:
    """
    Performs early existing using thresholds on the maximum predicted prob.

    Args:
        val_probs: A [C, L, K] array of predicted probabilities on the validation set
        test_probs: A [B, L, K] array of predicted probabilities on the test set
        rates: A [L] array of stopping rates (must sum to 1)
    Returns:
        A [B] array of predictions and a [B] array of the output indices for each
        prediction.
    """
    num_samples, num_outputs, num_classes = test_probs.shape

    assert len(rates) == num_outputs, 'Must provide {} rates. Got: {}'.format(len(rates), num_outputs)
    assert np.isclose(np.sum(rates), 1.0), 'Rates must sum to 1'
    assert num_outputs == 2, 'Only supports 2 outputs'

    # Compute the thresholds. For now, we use the given data to get 'perfect' rates.
    max_val_probs = np.max(val_probs, axis=-1)  # [B, L]

    t0 = np.quantile(max_val_probs[:, 0], q=1.0 - rates[0])
    thresholds = [t0, 0.0]

    # Get the predictions
    predictions: List[int] = []
    output_counts: List[int] = []

    test_preds = np.argmax(test_probs, axis=-1)

    for sample_idx in range(num_samples):
        sample_max_probs = np.max(test_probs[sample_idx], axis=-1)
        stop_comparison = np.greater(sample_max_probs, thresholds)
        stopped_idx = np.argmax(stop_comparison)
        pred = test_preds[sample_idx, stopped_idx]

        predictions.append(pred)
        output_counts.append(stopped_idx)

    return EarlyExitResult(preds=np.vstack(predictions).reshape(-1),
                           output_counts=np.vstack(output_counts).reshape(-1))

def even_max_prob_exit(val_probs: np.ndarray, test_probs: np.ndarray, val_labels: np.ndarray, rates: List[float], rand: np.random.RandomState, use_rand: bool) -> EarlyExitResult:
    """
    Performs early existing using thresholds on the maximum predicted prob.

    Args:
        val_probs: A [C, L, K] array of predicted probabilities on the validation set
        test_probs: A [B, L, K] array of predicted probabilities on the test set
        rates: A [L] array of stopping rates (must sum to 1)
    Returns:
        A [B] array of predictions and a [B] array of the output indices for each
        prediction.
    """
    num_samples, num_outputs, num_classes = test_probs.shape

    assert len(rates) == num_outputs, 'Must provide {} rates. Got: {}'.format(len(rates), num_outputs)
    assert np.isclose(np.sum(rates), 1.0), 'Rates must sum to 1'
    assert num_outputs == 2, 'Only supports 2 outputs'

    # Compute the thresholds. For now, we use the given data to get 'perfect' rates.
    max_val_probs = np.max(val_probs, axis=-1)  # [B, L]
    first_val_preds = np.argmax(val_probs[:, 0, :], axis=-1)  # [B]

    # Get the max prob for each prediction of the first output
    pred_distributions: DefaultDict[int, List[float]] = defaultdict(list)
    for sample_idx in range(val_probs.shape[0]):
        max_prob = max_val_probs[sample_idx, 0]
        pred = first_val_preds[sample_idx]
        pred_distributions[pred].append(max_prob)

    # Set the thresholds according to the percentile in each prediction
    class_thresholds: Dict[int, float] = dict()
    for pred, distribution in pred_distributions.items():
        t = np.quantile(distribution, q=1.0 - rates[0])
        class_thresholds[pred] = t

    # Get the "correct" rates for each prediction from the first level
    correct_counts = np.zeros((num_classes, ))
    total_counts = np.zeros((num_classes, ))

    for pred, label in zip(first_val_preds, val_labels):
        correct_counts[pred] += int(pred == label)
        total_counts[pred] += 1

    correct_rates = correct_counts / total_counts
    incorrect_rates = 1.0 - correct_rates

    # Get the predictions
    predictions: List[int] = []
    output_counts: List[int] = []

    test_preds = np.argmax(test_probs, axis=-1)
    test_max_probs = np.max(test_probs, axis=-1)

    for sample_idx in range(num_samples):
        r = rand.uniform()
        first_pred = test_preds[sample_idx, 0]
        rand_rate = incorrect_rates[first_pred]

        if (r < rand_rate) and use_rand:
            r_level = rand.uniform()
            stopped_idx = int(r_level >= rates[0])
        else:
            sample_thresholds = [class_thresholds[first_pred], 0.0]

            sample_max_probs = test_max_probs[sample_idx]
            stop_comparison = np.greater(sample_max_probs, sample_thresholds)
            stopped_idx = np.argmax(stop_comparison)

        pred = test_preds[sample_idx, stopped_idx]

        predictions.append(pred)
        output_counts.append(stopped_idx)

    return EarlyExitResult(preds=np.vstack(predictions).reshape(-1),
                           output_counts=np.vstack(output_counts).reshape(-1))
