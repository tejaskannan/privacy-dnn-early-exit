import numpy as np
import os.path
from annoy import AnnoyIndex
from typing import List, Tuple

from privddnn.classifier import BaseClassifier, OpName
from privddnn.dataset import Dataset


class NearestNeighborIterator:

    def __init__(self, dataset: Dataset, clf: BaseClassifier, window_size: int, num_trials: int, fold: str):
        self._clf = clf
        self._dataset = dataset
        self._idx = 0
        self._window_size = window_size
        self._num_trials = num_trials

        if fold == 'val':
            self._data_fold = dataset.get_val_inputs()
            self._probs = clf.validate(op=OpName.PROBS)
            self._labels = dataset.get_val_labels()
        elif fold == 'test':
            self._data_fold = dataset.get_test_inputs()
            self._probs = clf.test(op=OpName.PROBS)
            self._labels = dataset.get_test_labels()
        else:
            raise ValueError('Unknown fold with name {}'.format(fold))

        # Load the annoy index
        self._knn_index = AnnoyIndex(dataset.num_features, 'euclidean')
        dir_base = os.path.dirname(__file__)
        self._knn_index.load(os.path.join(dir_base, '..', 'data', dataset.dataset_name, '{}.ann'.format(fold)))

        # Initialize the current window
        self._current_window: List[int] = []
        self._window_idx = 0
        self._rand = np.random.RandomState(seed=20094823)

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def num_samples(self) -> int:
        return len(self._data_fold)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, int]:
        if self._idx >= (self.num_samples * self._num_trials):
            raise StopIteration

        # Create a new nearest-neighbor window
        if (self._window_idx >= self.window_size) or (self._idx == 0):
            base_idx = self._rand.randint(low=0, high=self.num_samples)
            self._current_window = self._knn_index.get_nns_by_item(base_idx, n=self.window_size)
            self._window_idx = 0

            window_indices = list(range(self.window_size))
            self._rand.shuffle(window_indices)
            shuffled_window = [self._current_window[i] for i in window_indices]
            self._current_window = shuffled_window

        data_idx = self._current_window[self._window_idx]

        self._idx += 1
        self._window_idx += 1

        return self._probs[data_idx], self._labels[data_idx]
