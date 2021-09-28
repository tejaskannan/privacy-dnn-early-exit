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

    log_ratio = np.log(probs_x / (probs_y + SMALL_NUMBER) + SMALL_NUMBER)
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


def compute_mutual_info(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
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

    return entropy_x + entropy_y - entropy_xy


def softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    """
    Computes the softmax function along the given axis.
    """
    max_logit = np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits - max_logit)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


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
