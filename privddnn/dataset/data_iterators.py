import numpy as np
import os.path
from annoy import AnnoyIndex
from typing import List, Tuple, Optional, Dict, Any

from privddnn.classifier import BaseClassifier, OpName
from privddnn.dataset import Dataset
from privddnn.dataset.build_nearest_neighbor_index import create_index


class DataIterator:

    def __init__(self, dataset: Dataset, clf: Optional[BaseClassifier], num_trials: int, fold: str):
        assert num_trials >= 1, 'Must provide a positive number of trials.'

        self._dataset = dataset
        self._idx = 0
        self._num_trials = num_trials
        self._clf = clf

        if fold == 'val':
            self._data_fold = dataset.get_val_inputs()
            self._labels = dataset.get_val_labels()
            self._probs = clf.validate(op=OpName.PROBS) if (clf is not None) else None
        elif fold == 'test':
            self._data_fold = dataset.get_test_inputs()
            self._labels = dataset.get_test_labels()
            self._probs = clf.test(op=OpName.PROBS) if (clf is not None) else None
        else:
            raise ValueError('Iterator does not support fold: {}'.format(fold))

        self._rand = np.random.RandomState(seed=20094823)
        self._num_labels = dataset.num_labels

    @property
    def num_samples(self) -> int:
        return len(self._data_fold)

    @property
    def num_trials(self) -> int:
        return self._num_trials

    @property
    def name(self) -> str:
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
        raise NotImplementedError()


class OriginalOrderIterator(DataIterator):

    def __init__(self, dataset: Dataset, clf: Optional[BaseClassifier], num_trials: int, fold: str):
        super().__init__(dataset=dataset, num_trials=num_trials, fold=fold, clf=clf)
        self._sample_idx = np.arange(len(self._data_fold))
        self._idx = 0

    @property
    def name(self) -> str:
        return 'original'

    def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
        if (self._idx >= self.num_samples * self.num_trials):
            raise StopIteration

        data_idx = self._idx % self.num_samples
        sample_probs = self._probs[data_idx] if (self._probs is not None) else None
        inputs = self._data_fold[data_idx]
        label = self._labels[data_idx]

        self._idx += 1
        return inputs, sample_probs, label


class RandomizedIterator(DataIterator):

    def __init__(self, dataset: Dataset, clf: Optional[BaseClassifier], num_trials: int, fold: str):
        super().__init__(dataset=dataset, num_trials=num_trials, fold=fold, clf=clf)
        self._sample_idx = np.arange(len(self._data_fold))
        self._rand.shuffle(self._sample_idx)
        self._idx = 0

    @property
    def name(self) -> str:
        return 'randomized'

    def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
        if (self._idx >= self.num_samples * self.num_trials):
            raise StopIteration

        data_idx = self._sample_idx[self._idx % self.num_samples]

        sample_probs = self._probs[data_idx] if (self._probs is not None) else None
        inputs = self._data_fold[data_idx]
        label = self._labels[data_idx]

        self._idx += 1
        return inputs, sample_probs, label


class NearestNeighborIterator(DataIterator):

    def __init__(self, dataset: Dataset, clf: Optional[BaseClassifier], window_size: int, num_trials: int, fold: str):
        assert window_size >= 1, 'Must provide a positive window size.'

        super().__init__(dataset=dataset, num_trials=num_trials, fold=fold, clf=clf)

        # Load the annoy index. This block builds the index if not already present.
        dir_base = os.path.dirname(__file__)
        index_path = os.path.join(dir_base, '..', 'data', dataset.dataset_name, '{}.ann'.format(fold))

        if not os.path.exists(index_path):
            create_index(inputs=self._data_fold, path=index_path)

        self._knn_index = AnnoyIndex(dataset.num_features, 'euclidean')
        self._knn_index.load(index_path)

        # Initialize the current window parameters
        self._window_size = window_size
        self._current_window: List[int] = []
        self._window_idx = 0

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def name(self) -> str:
        return 'nearest-{}'.format(self.window_size)

    def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
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

        sample_probs = self._probs[data_idx] if (self._probs is not None) else None
        return self._data_fold[data_idx], sample_probs, self._labels[data_idx]


def make_data_iterator(name: str, dataset: Dataset, clf: Optional[BaseClassifier], num_trials: int, fold: str, **kwargs: Dict[str, Any]) -> DataIterator:
    name = name.lower()

    if name in ('random', 'randomized'):
        return RandomizedIterator(dataset=dataset, clf=clf, num_trials=num_trials, fold=fold)
    elif name in ('original', 'original-order', 'original_order'):
        return OriginalOrderIterator(dataset=dataset, clf=clf, num_trials=num_trials, fold=fold)
    elif name in ('nearest', 'nearest_neighbor', 'nearest-neighbor'):
        assert kwargs.get('window_size') is not None, 'Must provide a window size.'
        return NearestNeighborIterator(dataset=dataset, clf=clf, num_trials=num_trials, fold=fold, window_size=int(kwargs['window_size']))
    else:
        raise ValueError('Unknown iterator for name: {}'.format(name))
