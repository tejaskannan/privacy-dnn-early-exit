import tensorflow as tf2
from .constants import SMALL_NUMBER


def compute_average_rates_per_label(rates: tf2.Tensor, labels: tf2.Tensor) -> tf2.Tensor:
    """
    Stratifies the predicted stop rates by class and computes the average.

    Args:
        rates: A [B, K] tensor of stop probabilities for each output (K)
        labels: A [B] tensor of the labels for each batch sample (B)
    Returns:
        A [L, K] tensor with the average rate per class
    """
    num_labels = tf2.reduce_max(labels) + 1
    average_rates = tf2.math.unsorted_segment_mean(data=rates,
                                                   segment_ids=labels,
                                                   num_segments=num_labels)
    return average_rates


def tf_compute_entropy(probs: tf2.Tensor, axis: int) -> tf2.Tensor:
    log_values = tf2.math.log(probs + SMALL_NUMBER)
    return -1 * tf2.reduce_sum(log_values * probs, axis=-1)


def make_max_prob_targets(labels: tf2.Tensor, num_labels: int, target_prob: tf2.Tensor) -> tf2.Tensor:
    """
    Creates probability distributions where the given label is the target probability and
    the other classses are uniform s.t. the result adds to 1.
    """
    label_space = tf2.expand_dims(tf2.range(start=0, limit=num_labels), axis=0)  # [1, L]
    labels = tf2.expand_dims(labels, axis=-1)  # [B, 1]

    target_prob = tf2.reshape(target_prob, (1, -1, 1))  # [1, K, 1]
    other_prob = (1.0 - target_prob) / (num_labels - 1)  # [1, K, 1]
    
    comparison = tf2.equal(labels, label_space)  # [B, L]
    comparison = tf2.expand_dims(comparison, axis=1)  # [B, 1, L]

    result = tf2.where(comparison, target_prob, other_prob)  # [B, K, L]
    return result
