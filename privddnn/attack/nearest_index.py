import numpy as np
from annoy import AnnoyIndex
from collections import Counter
from typing import Tuple, List

from privddnn.utils.file_utils import read_pickle_gz


def make_similar_attack_dataset(levels: np.ndarray, labels: List[int], window_size: int, num_samples: int, path: str, rand: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    input_list: List[int] = []
    output_list: List[int] = []

    sample_idx = rand.choice(np.arange(len(labels)), size=num_samples, replace=False)
    nearest = NearestIndex(levels=levels, labels=labels, path=path)

    for idx in sample_idx:
        label, level_count = nearest.get_neighbors(idx=idx, count=window_size)
        input_list.append(level_count)
        output_list.append(label)

    return np.vstack(input_list).reshape(-1), np.vstack(output_list).reshape(-1)


class NearestIndex:

    def __init__(self, levels: np.ndarray, labels: List[int], path: str):
        # Load the metadata
        metadata = read_pickle_gz('{}.pkl.gz'.format(path))
        num_features = metadata['n_components']

        # Load the annoy index
        self._index = AnnoyIndex(num_features, 'euclidean')
        self._index.load('{}.ann'.format(path))

        # Save reference to the levels and labels (given in the same order 
        # as the index was created)
        self._levels = levels
        self._labels = labels

    def get_neighbors(self, idx: int, count: int) -> Tuple[int, int]:
        """
        Returns the total number of levels and the majority label
        for the nearest neighbors of the given sample index.
        """
        nearest = self._index.get_nns_by_item(idx, count, search_k=-1, include_distances=False)

        levels = 0
        label_counter: Counter = Counter()

        for sample_idx in nearest:
            label = self._labels[sample_idx]
            level = self._levels[sample_idx]

            levels += level
            label_counter[label] += 1

        most_common = label_counter.most_common(1)

        return most_common[0][0], levels
