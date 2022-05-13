import numpy as np
from collections import Counter
from typing import List, Tuple


def create_ngrams(levels: List[int], preds: List[int], n: int, num_outputs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregates the levels and predictions into n-grams based on sequential level patterns.

    Args:
        levels: A list of output levels (ordered)
        preds: A list of predictions (ordered)
        n: The size of each pattern
    Returns:
        A tuple of (1) n-gram levels (index) and (2) majority prediction for each n-gram
    """
    assert len(levels) == len(preds), 'Must provide the same number of levels as predictions'

    ngram_inputs: List[int] = []
    ngram_outputs: List[int] = []
    base = num_outputs

    for idx in range(0, len(levels) - n, n):
        sample_levels = levels[idx:idx+n]

        if len(sample_levels) < n:
            continue

        ngram_index = int(''.join(map(str, sample_levels)), base)

        # Get the majority prediction
        pred_counter: Counter = Counter()
        for pred in preds[idx:idx+n]:
            pred_counter[pred] += 1

        sample_pred = pred_counter.most_common(1)[0][0]

        ngram_inputs.append(ngram_index)
        ngram_outputs.append(sample_pred)

    return np.vstack(ngram_inputs).reshape(-1), np.vstack(ngram_outputs).reshape(-1)


def create_ngram_counts(levels: List[int], preds: List[int], n: int, num_outputs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregates the levels and predictions into order-invariant n-grams based on sequential level patterns.

    Args:
        levels: A list of output levels (ordered)
        preds: A list of predictions (ordered)
        n: The size of each pattern
    Returns:
        A tuple of (1) n-gram levels (index) and (2) majority prediction for each n-gram
    """
    assert len(levels) == len(preds), 'Must provide the same number of levels as predictions'

    ngram_inputs: List[int] = []
    ngram_outputs: List[int] = []

    base = n + 1

    for idx in range(0, len(levels) - n, n):
        sample_levels = levels[idx:idx+n]

        if len(sample_levels) < n:
            continue

        window_counts = np.zeros(shape=(num_outputs, ), dtype=int)  # [W]
        for ell in sample_levels:
            window_counts[ell] += 1

        window_count_str = ''.join(map(str, window_counts))
        encoded_counts = int(window_count_str, base)

        # Get the majority prediction
        pred_counter: Counter = Counter()
        for pred in preds[idx:idx+n]:
            pred_counter[pred] += 1

        sample_pred = pred_counter.most_common(1)[0][0]

        ngram_inputs.append(encoded_counts)
        ngram_outputs.append(sample_pred)

    return np.vstack(ngram_inputs).reshape(-1), np.vstack(ngram_outputs).reshape(-1)
