import numpy as np
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
from typing import List, Tuple, Set


def create_ngrams(levels: List[int], preds: List[int], n: int, num_outputs: int, num_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregates the levels and predictions into n-grams based on sequential level patterns.

    Args:
        levels: A list of output levels (ordered)
        preds: A list of predictions (ordered)
        n: The size of each pattern
        num_outputs: The number of outputs in the model
        num_clusters: The number of ngram clusters to make
    Returns:
        A tuple of (1) n-gram levels (index) and (2) majority prediction for each n-gram
    """
    assert len(levels) == len(preds), 'Must provide the same number of levels as predictions'

    ngram_inputs: List[int] = []
    ngram_features_list: List[np.ndarray] = []
    ngram_outputs: List[int] = []
    base = num_outputs

    unique_ngrams: Set[str] = set()

    for idx in range(0, len(levels), n):
        sample_levels = levels[idx:idx+n]

        if len(sample_levels) < n:
            continue

        # Make the feature vector
        features = np.zeros(shape=(n, num_outputs))
        for ngram_idx, level in enumerate(sample_levels):
            features[ngram_idx, level] = 1

        ngram_string = ''.join(map(str, sample_levels))
        ngram_index = int(ngram_string, base)

        unique_ngrams.add(ngram_string)

        # Get the majority prediction
        pred_counter: Counter = Counter()
        for pred in preds[idx:idx+n]:
            pred_counter[pred] += 1

        sample_pred = pred_counter.most_common(1)[0][0]

        ngram_inputs.append(ngram_index)
        ngram_features_list.append(features.astype(float).reshape(1, -1))
        ngram_outputs.append(sample_pred)

    ngram_outputs = np.vstack(ngram_outputs).reshape(-1)

    if (len(ngram_inputs) < num_clusters) or (len(unique_ngrams) < num_clusters):
        return np.vstack(ngram_inputs).reshape(-1), ngram_outputs

    ngram_features = np.vstack(ngram_features_list)
    kmeans = MiniBatchKMeans(n_clusters=num_clusters,
                             random_state=43289)

    ngram_inputs = kmeans.fit_predict(ngram_features).astype(int)

    return ngram_inputs, ngram_outputs


def create_ngram_counts(levels: List[int], preds: List[int], n: int, num_outputs: int, num_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregates the levels and predictions into order-invariant n-grams based on sequential level patterns.

    Args:
        levels: A list of output levels (ordered)
        preds: A list of predictions (ordered)
        n: The size of each pattern
        num_outputs: The true number of model outputs
        num_clusters: The number of clusters to group ngrams into
    Returns:
        A tuple of (1) n-gram levels (index) and (2) majority prediction for each n-gram
    """
    assert len(levels) == len(preds), 'Must provide the same number of levels as predictions'

    ngram_inputs: List[int] = []
    ngram_features_list: List[np.ndarray] = []
    ngram_outputs: List[int] = []

    base = n + 1
    unique_ngrams: Set[str] = set()

    for idx in range(0, len(levels), n):
        sample_levels = levels[idx:idx+n]

        if len(sample_levels) < n:
            continue

        window_counts = np.zeros(shape=(num_outputs, ), dtype=int)  # [W]
        for ell in sample_levels:
            window_counts[ell] += 1

        window_count_str = ''.join(map(str, window_counts))
        encoded_counts = int(window_count_str, base)

        unique_ngrams.add(window_count_str)

        # Get the majority prediction
        pred_counter: Counter = Counter()
        for pred in preds[idx:idx+n]:
            pred_counter[pred] += 1

        sample_pred = pred_counter.most_common(1)[0][0]

        ngram_features_list.append(np.expand_dims(window_counts, axis=0))
        ngram_inputs.append(encoded_counts)
        ngram_outputs.append(sample_pred)

    ngram_outputs = np.vstack(ngram_outputs).reshape(-1)

    if len(unique_ngrams) < num_clusters:
        return np.vstack(ngram_inputs).reshape(-1), ngram_outputs

    ngram_features = np.vstack(ngram_features_list)
    kmeans = KMeans(n_clusters=num_clusters, random_state=932)
    ngram_inputs = kmeans.fit_predict(ngram_features)

    return ngram_inputs, ngram_outputs
