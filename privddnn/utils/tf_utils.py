import tensorflow as tf2


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
