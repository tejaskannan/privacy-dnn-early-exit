import numpy as np
from annoy import AnnoyIndex
from collections import Counter, defaultdict
from typing import Tuple, List, DefaultDict

from privddnn.utils.file_utils import read_pickle_gz


def make_noisy_dataset(levels: List[int],
                       preds: List[int],
                       window_size: int,
                       num_samples: int,
                       noise_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    assert len(levels) == len(preds), 'Levels ({}) and Preds ({}) are misaligned.'.format(len(levels), len(preds))

    level_dist: DefaultDict[int, List[int]] = defaultdict(list)
    for level, pred in zip(levels, preds):
        level_dist[pred].append(level)

    num_labels = len(level_dist)

    rand = np.random.RandomState(seed=8996083)
    sample_idx = np.arange(len(levels))
    selected_idx = rand.choice(sample_idx, size=num_samples, replace=True)

    input_list: List[np.ndarray] = []
    output_list: List[np.ndarray] = []

    # The number of elements per window that come from any label
    num_noise = int(noise_rate * window_size)

    for sample_idx in selected_idx:
        # Derive the input features from the given sample index
        pred = preds[sample_idx]
        level_counts = level_dist[pred]

        # Create the input sample for the selected label
        selected_levels = rand.choice(level_counts, size=window_size - num_noise, replace=True)
        noise_levels = rand.choice(levels, size=num_noise, replace=True)

        # Create the features by combining the correct and noisy values and then shuffling
        input_features = np.concatenate([selected_levels, noise_levels], axis=0)  # [W]
        rand.shuffle(input_features)

        input_list.append(np.expand_dims(input_features, axis=0))
        output_list.append(pred)

    return np.vstack(input_list), np.vstack(output_list).reshape(-1)


def make_sequential_dataset(levels: List[int], preds: List[int], window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    input_list: List[np.ndarray] = []
    output_list: List[int] = []

    for idx in range(0, len(preds), window_size):
        sample_levels = levels[idx:idx+window_size]

        if len(sample_levels) < window_size:
            continue

        # Get the majority prediction
        pred_counter: Counter = Counter()
        for pred in preds[idx:idx+window_size]:
            pred_counter[pred] += 1

        sample_pred = pred_counter.most_common(1)[0][0]
        
        input_list.append(np.expand_dims(sample_levels, axis=0))
        output_list.append(sample_pred)

    inputs = np.vstack(input_list)
    inputs = 2 * inputs - 1  # Translate into +1, -1 values

    return inputs, np.vstack(output_list).reshape(-1)


def make_similar_dataset(levels: List[int],
                         preds: List[int],
                         window_size: int,
                         path: str) -> Tuple[np.ndarray, np.ndarray]:
    assert len(levels) == len(preds), 'Levels ({}) and Preds ({}) are misaligned.'.format(len(levels), len(preds))

    input_list: List[int] = []
    output_list: List[int] = []
    nearest = NearestIndex(levels=levels, preds=preds, path=path)

    for idx in range(len(preds)):
        sample_pred, sample_levels = nearest.get_neighbors(idx=idx, count=window_size)

        input_list.append(np.expand_dims(sample_levels, axis=0))
        output_list.append(sample_pred)

    return np.vstack(input_list), np.vstack(output_list).reshape(-1)


class NearestIndex:

    def __init__(self, levels: List[int], preds: List[int], path: str):
        # Load the metadata
        metadata = read_pickle_gz('{}.pkl.gz'.format(path))
        num_features = metadata['n_components']

        # Load the annoy index
        self._index = AnnoyIndex(num_features, 'euclidean')
        self._index.load('{}.ann'.format(path))

        # Save reference to the levels and labels (given in the same order 
        # as the index was created)
        self._levels = levels
        self._preds = preds

    def get_neighbors(self, idx: int, count: int) -> Tuple[int, np.ndarray]:
        """
        Returns the total number of levels and the majority label
        for the nearest neighbors of the given sample index.
        """
        nearest = self._index.get_nns_by_item(idx, count, search_k=-1, include_distances=False)

        levels: List[int] = []
        pred_counter: Counter = Counter()

        for sample_idx in nearest:
            pred = self._preds[sample_idx]
            level = self._levels[sample_idx]

            levels.append(level)
            pred_counter[label] += 1

        most_common = pred_counter.most_common(1)
        return most_common[0][0], np.vstack(levels).reshape(-1)
