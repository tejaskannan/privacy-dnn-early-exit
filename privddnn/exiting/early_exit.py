import numpy as np
from collections import namedtuple, defaultdict
from typing import List, Tuple, Dict, DefaultDict

from privddnn.utils.metrics import compute_entropy
from privddnn.utils.constants import BIG_NUMBER


EarlyExitResult = namedtuple('EarlyExitResult', ['preds', 'output_counts'])


def random_exit(probs: np.ndarray, rates: List[float], rand: np.random.RandomState) -> EarlyExitResult:
    """
    Performs early-exiting random of the input label.

    Args:
        probs: A [B, L, K] array of predicted probabilities
        rates: A [L] list of stopping rates (must sum to 1)
        rand: The random state used for determining stopping with reproducible results
    Returns:
        A [B] array of predictions based on the stopping criteria and a [B] array
        of the output indices for each prediction
    """
    num_samples, num_outputs, num_classes = probs.shape

    assert len(rates) == num_outputs, 'Must provide {} rates. Got: {}'.format(len(rates), num_outputs)
    assert np.isclose(np.sum(rates), 1.0), 'Rates must sum to 1'

    predictions: List[int] = []
    output_counts: List[int] = []

    for sample_probs in probs:
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


def entropy_exit(probs: np.ndarray, rates: List[float]) -> EarlyExitResult:
    """
    Performs early existing using thresholds on entropy of the predicted probs.

    Args:
        probs: A [B, L, K] array of predicted probabilities
        rates: A [L] array of stopping rates (must sum to 1)
    Returns:
        A [B] array of predictions and a [B] array of the output indices for each
        prediction.
    """
    num_samples, num_outputs, num_classes = probs.shape

    assert len(rates) == num_outputs, 'Must provide {} rates. Got: {}'.format(len(rates), num_outputs)
    assert np.isclose(np.sum(rates), 1.0), 'Rates must sum to 1'
    assert num_outputs == 2, 'Only supports 2 outputs'

    # Compute the thresholds. For now, we use the given data to get 'perfect' rates.
    pred_entropy = compute_entropy(probs, axis=-1)  # [B, L]

    t0 = np.quantile(pred_entropy[:, 0], q=rates[0])
    thresholds = [t0, BIG_NUMBER]

    # Get the predictions
    predictions: List[int] = []
    output_counts: List[int] = []

    for sample_idx in range(num_samples):
        sample_entropy = pred_entropy[sample_idx]  # [L]
        stop_comparison = np.less(sample_entropy, thresholds)
        stopped_idx = np.argmax(stop_comparison)
        pred = np.argmax(probs[sample_idx, stopped_idx])

        predictions.append(pred)
        output_counts.append(stopped_idx)

    return EarlyExitResult(preds=np.vstack(predictions).reshape(-1),
                           output_counts=np.vstack(output_counts).reshape(-1))


def max_prob_exit(probs: np.ndarray, rates: List[float]) -> EarlyExitResult:
    """
    Performs early existing using thresholds on the maximum predicted prob.

    Args:
        probs: A [B, L, K] array of predicted probabilities
        rates: A [L] array of stopping rates (must sum to 1)
    Returns:
        A [B] array of predictions and a [B] array of the output indices for each
        prediction.
    """
    num_samples, num_outputs, num_classes = probs.shape

    assert len(rates) == num_outputs, 'Must provide {} rates. Got: {}'.format(len(rates), num_outputs)
    assert np.isclose(np.sum(rates), 1.0), 'Rates must sum to 1'
    assert num_outputs == 2, 'Only supports 2 outputs'

    # Compute the thresholds. For now, we use the given data to get 'perfect' rates.
    max_probs = np.max(probs, axis=-1)  # [B, L]

    t0 = np.quantile(max_probs[:, 0], q=1.0 - rates[0])
    thresholds = [t0, 0.0]

    # Get the predictions
    predictions: List[int] = []
    output_counts: List[int] = []

    for sample_idx in range(num_samples):
        sample_max_probs = np.max(probs[sample_idx], axis=-1)
        stop_comparison = np.greater(sample_max_probs, thresholds)
        stopped_idx = np.argmax(stop_comparison)
        pred = np.argmax(probs[sample_idx, stopped_idx])

        predictions.append(pred)
        output_counts.append(stopped_idx)

    return EarlyExitResult(preds=np.vstack(predictions).reshape(-1),
                           output_counts=np.vstack(output_counts).reshape(-1))

def even_max_prob_exit(probs: np.ndarray, labels: np.ndarray, rates: List[float]) -> EarlyExitResult:
    """
    Performs early existing using thresholds on the maximum predicted prob.

    Args:
        probs: A [B, L, K] array of predicted probabilities
        rates: A [L] array of stopping rates (must sum to 1)
    Returns:
        A [B] array of predictions and a [B] array of the output indices for each
        prediction.
    """
    num_samples, num_outputs, num_classes = probs.shape

    assert len(rates) == num_outputs, 'Must provide {} rates. Got: {}'.format(len(rates), num_outputs)
    assert np.isclose(np.sum(rates), 1.0), 'Rates must sum to 1'
    assert num_outputs == 2, 'Only supports 2 outputs'

    # Compute the thresholds. For now, we use the given data to get 'perfect' rates.
    max_probs = np.max(probs, axis=-1)  # [B, L]

    # Get the max prob for each prediction of the first output
    pred_distributions: DefaultDict[int, List[float]] = defaultdict(list)
    for sample_idx in range(num_samples):
        sample_label = labels[sample_idx]
        max_prob = np.max(probs[sample_idx, 0])
        pred_distributions[sample_label].append(max_prob)

    # Set the thresholds according to the percentile in each prediction
    class_thresholds: Dict[int, float] = dict()
    for sample_label, distribution in pred_distributions.items():
        t = np.quantile(distribution, q=1.0 - rates[0])
        class_thresholds[sample_label] = t

    # Get the predictions
    predictions: List[int] = []
    output_counts: List[int] = []

    for sample_idx in range(num_samples):
        sample_label = labels[sample_idx]
        sample_thresholds = [class_thresholds[sample_label], 0.0]

        sample_max_probs = np.max(probs[sample_idx], axis=-1)
        stop_comparison = np.greater(sample_max_probs, sample_thresholds)

        stopped_idx = np.argmax(stop_comparison)
        pred = np.argmax(probs[sample_idx, stopped_idx])

        predictions.append(pred)
        output_counts.append(stopped_idx)

    return EarlyExitResult(preds=np.vstack(predictions).reshape(-1),
                           output_counts=np.vstack(output_counts).reshape(-1))
