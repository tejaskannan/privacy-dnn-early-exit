import numpy as np
import math
from collections import defaultdict
from typing import List, DefaultDict, Tuple, Union
from .constants import SMALL_NUMBER


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes the accuracy of the given predictions
    """
    assert len(predictions.shape) == len(labels.shape), 'Must provide same # dimensions for preds ({}) and labels ({})'.format(len(predictions.shape), len(labels.shape))

    correct = np.isclose(predictions, labels)
    return np.average(correct)


def create_confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Computes the L x L confusion matrix where [i][j] is the rate of elements predicted
    as i that are actually label j
    """
    assert len(predictions.shape) == 1, 'Must provide a 1d predictions array'
    assert predictions.shape == labels.shape, 'Must provide equal-sized predictions and labels'

    num_labels = np.max(labels) + 1
    confusion_mat = np.zeros(shape=(num_labels, num_labels), dtype=float)

    for pred, label in zip(predictions, labels):
        confusion_mat[pred, label] += 1.0

    return confusion_mat


def create_metric_distributions(predictions: np.ndarray, labels: np.ndarray, metrics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Comptutes the average and std deviation metric value condition on the (pred, label) pair
    """
    assert len(predictions.shape) == 1, 'Must provide a 1d predictions array'
    assert predictions.shape == labels.shape, 'Must provide equal-sized predictions and labels'
    assert predictions.shape == metrics.shape, 'Must provide equal-sized predictions and metrics'

    num_labels = np.max(labels) + 1
    distributions: DefaultDict[Tuple[int, int], List[float]] = defaultdict(list)

    for pred, label, metric in zip(predictions, labels, metrics):
        key = (pred, label)
        distributions[key].append(metric)

    means = np.zeros(shape=(num_labels, num_labels))
    stds = np.zeros_like(means)

    for (pred, label), metrics in distributions.items():
        if len(metrics) > 0:
            means[pred, label] = np.average(metrics)
            stds[pred, label] = np.std(metrics)

    return means, stds


def compute_entropy(probs: np.ndarray, axis: int) -> np.ndarray:
    """
    Computes the (empirical) entropy of the given distributions along the given axis.
    """
    log_probs = np.log2(probs + SMALL_NUMBER)
    return -1 * np.sum(probs * log_probs, axis=axis)


def get_joint_distribution(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Computes the (empirical) joint distribution  p(X, Y)
    """
    assert len(X.shape) == 1, 'Must provide a 1d input for X'
    assert len(Y.shape) == 1, 'Must provide a 1d input for Y'

    bins_x = np.max(X) + 1
    bins_y = np.max(Y) + 1

    joint_counts = np.histogram2d(X, Y, bins=[bins_x, bins_y])[0]
    joint_probs = joint_counts / np.sum(joint_counts)
    return joint_probs


def compute_kl_divergence(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Computes the (empirical) KL divergence between the samples from X and Y.
    This function returns D(X || Y).
    """
    # Validate the arguments
    assert np.max(X) == np.max(Y), 'X and Y must have the same maximum value'
    assert np.min(X) == np.min(Y), 'X and Y must have the same minimum value'

    joint_distribution = get_joint_distribution(X=X, Y=Y)

    probs_x = np.sum(joint_distribution, axis=-1)
    probs_y = np.sum(joint_distribution, axis=0)

    assert len(probs_x) == len(probs_y), 'P(x) and P(y) must have the same length'

    log_ratio = np.log2(probs_x / (probs_y + SMALL_NUMBER) + SMALL_NUMBER)
    return np.sum(probs_x * log_ratio)


def compute_conditional_entropy(joint_distribution: np.ndarray) -> np.ndarray:
    """
    Computes the (empirical) conditional entropy of the given joint distribution H(X|Y)
    where the joint distribution is (i, j) -> p(x, y)
    """
    assert len(joint_distribution.shape) == 2, 'Must provide a 2d joint distribution'

    # Marginalize over the X axis
    probs_y = np.sum(joint_distribution, axis=0)

    cond_entropy = 0.0
    for i in range(joint_distribution.shape[0]):
        for j in range(joint_distribution.shape[1]):
            joint_prob = joint_distribution[i, j]
            cond_entropy -= joint_prob * np.log2((joint_prob / (probs_y[j] + SMALL_NUMBER)) + SMALL_NUMBER)

    return cond_entropy


def compute_joint_entropy(joint_distribution: np.ndarray) -> np.ndarray:
    """
    Computes the joint entropy between X and Y
    """
    assert len(joint_distribution.shape) == 2, 'Must provide a 2d joint distribution'

    joint_entropy = 0.0
    for i in range(joint_distribution.shape[0]):
        for j in range(joint_distribution.shape[1]):
            joint_prob = joint_distribution[i, j]
            joint_entropy -= joint_prob * np.log2(joint_prob + SMALL_NUMBER)

    return joint_entropy


def compute_mutual_info(X: np.ndarray, Y: np.ndarray, should_normalize: bool) -> np.ndarray:
    """
    Computes the mutual information between the samples X and Y
    """
    assert len(X.shape) == 1, 'Must provide a 1d input for X'
    assert len(Y.shape) == 1, 'Must provide a 1d input for Y'

    joint_probs = get_joint_distribution(X=X, Y=Y)

    probs_x = np.sum(joint_probs, axis=1)
    probs_y = np.sum(joint_probs, axis=0)

    entropy_x = compute_entropy(probs_x, axis=0)
    entropy_y = compute_entropy(probs_y, axis=0)
    entropy_xy = compute_joint_entropy(joint_probs)

    mut_info = entropy_x + entropy_y - entropy_xy
    return (mut_info) / max(entropy_x, entropy_y) if should_normalize else mut_info


def softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    """
    Computes the softmax function along the given axis.
    """
    max_logit = np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits - max_logit)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-1 * x))


def linear_step(x: float, width: float, clip: float) -> float:
    if (x > ((width / 2.0) - 1.0)):
        return clip
    elif (x < -1 * ((width / 2.0) - 1.0)):
        return -1 * clip
    else:
        return 0.0
    #elif (x >= -1 * (width / 4.0)) and (x <= (width / 4.0)):
    #    return 0.0
    #else:
    #    slope = (4.0 * clip) / width
    #    return slope * x


def compute_max_prob_metric(probs: np.ndarray) -> np.ndarray:
    return np.max(probs, axis=-1)


def compute_entropy_metric(probs: np.ndarray) -> np.ndarray:
    num_labels = probs.shape[-1]
    uniform_dist = np.ones(shape=(num_labels, )) / num_labels
    max_entropy = float(compute_entropy(uniform_dist, axis=-1))
    return ((-1 * compute_entropy(probs, axis=-1)) / max_entropy) + 1.0


def compute_target_exit_rates(probs: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """
    Gets the fraction of elements for each prediction
    which exit early (at each level).

    Args:
        probs: A [B, K, L] array of predicted probabilities for each sample (B)
            and output level (K)
    Returns:
        The frequency at which each label (L) stop at each level (K). Each column
        in the [K, L] result array sums to 1.
    """
    assert len(probs.shape) == 3, 'Must provide a 3d array'

    # Unpack the shape
    num_samples, num_levels, num_labels = probs.shape
    assert num_levels == 2, 'Only supports 2-level situations'
    assert len(rates) == num_levels, 'Number of rates must equal number of levels'

    preds = np.argmax(probs, axis=-1)  # [B, K]
    pred_counts = np.zeros(shape=(num_levels, num_labels), dtype=float)  # [K, L]

    # Execute a bincount 2d
    for idx in range(num_samples):
        for level, pred in enumerate(preds[idx]):
            pred_counts[level, pred] += 1.0

    # Normalize over the label counts in a weighted manner
    rates = np.expand_dims(rates, axis=-1)  # [K, 1]
    weighted_counts = pred_counts * rates  # [K, L]
    pred_freq = weighted_counts / (np.sum(weighted_counts, axis=0, keepdims=True) + SMALL_NUMBER)

    return pred_freq


def compute_stop_counts(probs: np.ndarray) -> np.ndarray:
    """
    Computes the largest stop rate possible using the given validation probabilities.

    Args:
        probs: A [B, K, L] array of predicted probabilities for each sample (B) and output level (K).
    Returns:
        A [L] array of the maximum stop rates for each (predicted) label (L)
    """
    assert len(probs.shape) == 3, 'Must provide a 3d array.'
    assert probs.shape[1] == 2, 'Computation only works on 2-level models.'

    # Unpack the shape
    num_samples, _, num_labels = probs.shape

    stay_counts = np.zeros(shape=(num_labels, ))  # [L]
    elev_counts = np.zeros_like(stay_counts)  # [L]

    preds = np.argmax(probs, axis=-1)  # [B, K]

    for sample_idx in range(num_samples):
        first_pred, second_pred = preds[sample_idx, 0], preds[sample_idx, 1]

        stay_counts[first_pred] += 1

        if first_pred != second_pred:
            elev_counts[second_pred] += 1

    return np.vstack([np.expand_dims(stay_counts, axis=0), np.expand_dims(elev_counts, axis=0)])
    #max_stop_rates = stay_counts / (stay_counts + elev_counts + SMALL_NUMBER)
    #return max_stop_rates


def compute_avg_level_per_class(output_levels: List[int], labels: List[int]) -> List[float]:
    """
    Computes the average output level between 0 and 1 stratified by the label.
    """
    num_labels = max(labels) + 1
    result: List[float] = []

    for label in range(num_labels):
        mask = np.equal(labels, label).astype(int)
        masked_levels = np.multiply(output_levels, mask).astype(int)

        levels_sum = np.sum(masked_levels)
        label_count = np.sum(mask)
        avg_level = levels_sum / label_count
        result.append(avg_level)

    return result


def compute_max_likelihood_ratio(label_stop_probs: List[float], window_size: int) -> float:
    """
    Gets the maximum likelihood ratio across all counts assuming each stopping pattern
    follows bernoulli trials with the given stopping probabilties (for each label).
    """
    num_labels = len(label_stop_probs)
    highest_ratio = 0.0

    for label1 in range(num_labels):

        weighted_ratios: List[float] = []

        for n in range(window_size):

            prob1 = label_stop_probs[label1]
            w_choose_n = math.factorial(window_size) / (math.factorial(window_size - n) * math.factorial(n))
            p1 = np.power(prob1, n) * np.power(1.0 - prob1, window_size - n)
            prob_n = w_choose_n * p1

            comp_ratio = 0.0  # Highest ratio over all OTHER labels

            for label2 in range(label1 + 1, num_labels):
                prob2 = label_stop_probs[label2]
                p2 = np.power(prob2, n) * np.power(1.0 - prob2, window_size - n)

                ratio = 1.0 - p2 / p1
                comp_ratio = max(comp_ratio, ratio)

            weighted_ratios.append(prob_n * comp_ratio)

        highest_ratio = max(highest_ratio, np.sum(weighted_ratios))

    return highest_ratio


def compute_geometric_mean(x: Union[List[float], np.ndarray]) -> np.ndarray:
    assert all([val > 1e-4 for val in x]), 'All values must be > 1e-4'
    prod = np.prod(x)
    n = len(x)
    return np.power(prod, 1.0 / n)


def to_one_hot(y: np.ndarray, num_labels: int) -> np.ndarray:
    """
    Converts the 1d integer array of predicted / true labels
    to one-hot vectors.
    """
    assert len(y.shape) == 1, 'Must provide a 1d input'
    num_samples = y.shape[0]

    result = np.zeros(shape=(num_samples, num_labels), dtype=float)
    
    sample_idx = np.arange(num_samples)
    result[sample_idx, y] = 1.0
    return result
