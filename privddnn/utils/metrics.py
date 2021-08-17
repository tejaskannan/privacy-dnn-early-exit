import numpy as np
from .constants import SMALL_NUMBER


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes the accuracy of the given predictions
    """
    assert len(predictions.shape) == len(labels.shape), 'Must provide same # dimensions for preds ({}) and labels ({})'.format(len(predictions.shape), len(labels.shape))

    correct = np.isclose(predictions, labels)
    return np.average(correct)


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
