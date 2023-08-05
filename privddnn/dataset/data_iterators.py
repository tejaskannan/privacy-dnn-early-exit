import numpy as np
import os.path
from collections import defaultdict
from annoy import AnnoyIndex
from typing import List, Tuple, Optional, Dict, Any, DefaultDict

from privddnn.classifier import BaseClassifier, OpName
from privddnn.dataset import Dataset
from privddnn.dataset.build_nearest_neighbor_index import NUM_COMPONENTS
from privddnn.utils.constants import SMALL_NUMBER
from privddnn.utils.file_utils import read_pickle_gz


class DataIterator:

    def __init__(self, dataset: Dataset, pred_probs: Optional[np.ndarray], num_reps: int, fold: str):
        assert num_reps >= 1, 'Must provide a positive number of reps.'

        self._dataset = dataset
        self._idx = 0
        self._num_reps = num_reps
        self._probs = pred_probs

        if fold == 'val':
            self._data_fold = dataset.get_val_inputs()
            self._labels = dataset.get_val_labels()
        elif fold == 'test':
            self._data_fold = dataset.get_test_inputs()
            self._labels = dataset.get_test_labels()
        else:
            raise ValueError('Iterator does not support fold: {}'.format(fold))

        self._rand = np.random.RandomState(seed=20094823)
        self._num_labels = dataset.num_labels

    @property
    def num_samples(self) -> int:
        return len(self._data_fold)

    @property
    def num_reps(self) -> int:
        return self._num_reps

    @property
    def name(self) -> str:
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
        raise NotImplementedError()


class OriginalOrderIterator(DataIterator):

    def __init__(self, dataset: Dataset, pred_probs: Optional[np.ndarray], num_reps: int, fold: str):
        super().__init__(dataset=dataset, num_reps=num_reps, fold=fold, pred_probs=pred_probs)
        self._sample_idx = np.arange(len(self._data_fold))
        self._idx = 0

    @property
    def name(self) -> str:
        return 'original'

    def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
        if (self._idx >= self.num_samples * self.num_reps):
            raise StopIteration

        data_idx = self._idx % self.num_samples
        sample_probs = self._probs[data_idx] if (self._probs is not None) else None
        inputs = self._data_fold[data_idx]
        label = self._labels[data_idx]

        self._idx += 1
        return inputs, sample_probs, label


class RandomizedIterator(DataIterator):

    def __init__(self, dataset: Dataset, pred_probs: Optional[np.ndarray], num_reps: int, fold: str):
        super().__init__(dataset=dataset, num_reps=num_reps, fold=fold, pred_probs=pred_probs)
        self._sample_idx = np.arange(len(self._data_fold))
        self._rand.shuffle(self._sample_idx)
        self._idx = 0

    @property
    def name(self) -> str:
        return 'randomized'

    def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
        if (self._idx >= self.num_samples * self.num_reps):
            raise StopIteration

        data_idx = self._sample_idx[self._idx % self.num_samples]

        sample_probs = self._probs[data_idx] if (self._probs is not None) else None
        inputs = self._data_fold[data_idx]
        label = self._labels[data_idx]

        self._idx += 1
        return inputs, sample_probs, label


class NearestNeighborIterator(DataIterator):

    def __init__(self, dataset: Dataset, pred_probs: Optional[np.ndarray], window_size: int, num_reps: int, fold: str):
        assert window_size >= 1, 'Must provide a positive window size.'

        super().__init__(dataset=dataset, num_reps=num_reps, fold=fold, pred_probs=pred_probs)

        # Load the annoy index. This block builds the index if not already present.
        dir_base = os.path.dirname(__file__)
        index_path = os.path.join(dir_base, '..', 'data', dataset.dataset_name, '{}.ann'.format(fold))

        assert os.path.exists(index_path), 'Must create the nearest neighbor index at {} using the create index script.'.format(index_path)

        num_features = np.prod(self._data_fold.shape[1:])
        n_components = min(num_features, NUM_COMPONENTS)

        self._knn_index = AnnoyIndex(n_components, 'euclidean')
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
        if self._idx >= (self.num_samples * self._num_reps):
            raise StopIteration

        # Create a new nearest-neighbor window in the order of the nearest neighbors.
        if (self._window_idx >= self.window_size) or (self._idx == 0):
            base_idx = self._rand.randint(low=0, high=self.num_samples)
            self._current_window = self._knn_index.get_nns_by_item(base_idx, n=self.window_size)
            self._window_idx = 0

        data_idx = self._current_window[self._window_idx]

        self._idx += 1
        self._window_idx += 1

        sample_probs = self._probs[data_idx] if (self._probs is not None) else None
        return self._data_fold[data_idx], sample_probs, self._labels[data_idx]


class SameLabelIterator(DataIterator):

    def __init__(self, dataset: Dataset, pred_probs: Optional[np.ndarray], window_size: int, noise_rate: float, num_reps: int, fold: str):
        assert window_size >= 1, 'Must provide a positive window size.'

        super().__init__(dataset=dataset, num_reps=num_reps, fold=fold, pred_probs=pred_probs)

        # Initialize the current window parameters
        self._window_size = window_size
        self._current_window: List[int] = []
        self._window_idx = 0
        self._noise_rate = noise_rate

        # Get the indices that have the same label
        self._same_label_indices: DefaultDict[int, List[int]] = defaultdict(list)

        for idx, label in enumerate(self._labels):
            self._same_label_indices[label].append(idx)

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def noise_rate(self) -> float:
        return self._noise_rate

    @property
    def name(self) -> str:
        if abs(self._noise_rate - 0.2) < SMALL_NUMBER:
            return 'same-label-{}'.format(self.window_size)
        else:
            return 'same-label-{}-{}'.format(self.window_size, int(self.noise_rate * 100.0))

    def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
        if self._idx >= (self.num_samples * self._num_reps):
            raise StopIteration

        # Create a new nearest-neighbor window
        if (self._window_idx >= self.window_size) or (self._idx == 0):
            base_idx = self._rand.randint(low=0, high=self.num_samples)
            base_label = self._labels[base_idx]

            self._current_window: List[int] = []

            same_rate = (1.0 - self.noise_rate)
            same_label_idx = self._rand.choice(self._same_label_indices[base_label], size=int(same_rate * self._window_size))
            self._current_window.extend(same_label_idx)

            remaining_elements = self._window_size - int(same_rate * self._window_size)
            rand_idx = self._rand.randint(low=0, high=self.num_samples, size=remaining_elements)
            self._current_window.extend(rand_idx)

            self._rand.shuffle(self._current_window)
            self._window_idx = 0

        data_idx = self._current_window[self._window_idx]

        self._idx += 1
        self._window_idx += 1

        sample_probs = self._probs[data_idx] if (self._probs is not None) else None
        return self._data_fold[data_idx], sample_probs, self._labels[data_idx]


class SameDataIterator(DataIterator):

    def __init__(self, dataset: Dataset, pred_probs: Optional[np.ndarray], window_size: int, num_reps: int, fold: str):
        assert window_size >= 1, 'Must provide a positive window size.'

        super().__init__(dataset=dataset, num_reps=num_reps, fold=fold, pred_probs=pred_probs)

        # Initialize the current window parameters
        self._window_size = window_size
        self._current_window: List[int] = []
        self._window_idx = 0
        self._base_idx = 0

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def name(self) -> str:
        return 'same-data-{}'.format(self.window_size)

    def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
        if self._idx >= (self.num_samples * self._num_reps):
            raise StopIteration

        # Create a new nearest-neighbor window
        if (self._window_idx >= self.window_size) or (self._idx == 0):
            self._base_idx = self._rand.randint(low=0, high=self.num_samples)
            self._window_idx = 0

        self._idx += 1
        self._window_idx += 1

        sample_probs = self._probs[self._base_idx] if (self._probs is not None) else None
        return self._data_fold[self._base_idx], sample_probs, self._labels[self._base_idx]


def make_data_iterator(name: str, dataset: Dataset, pred_probs: Optional[np.ndarray], num_reps: int, fold: str, **kwargs: Dict[str, Any]) -> DataIterator:
    name = name.lower()

    if name in ('random', 'randomized'):
        return RandomizedIterator(dataset=dataset, pred_probs=pred_probs, num_reps=num_reps, fold=fold)
    elif name in ('original', 'original-order', 'original_order'):
        return OriginalOrderIterator(dataset=dataset, pred_probs=pred_probs, num_reps=num_reps, fold=fold)
    elif name in ('nearest', 'nearest_neighbor', 'nearest-neighbor'):
        assert kwargs.get('window_size') is not None, 'Must provide a window size.'
        return NearestNeighborIterator(dataset=dataset, pred_probs=pred_probs, num_reps=num_reps, fold=fold, window_size=int(kwargs['window_size']))
    elif name in ('same-label', 'same_label'):
        assert kwargs.get('window_size') is not None, 'Must provide a window size.'
        return SameLabelIterator(dataset=dataset, pred_probs=pred_probs, num_reps=num_reps, fold=fold, window_size=int(kwargs['window_size']), noise_rate=float(kwargs.get('noise_rate', 0.2)))
    elif name in ('same-data', 'same_data'):
        assert kwargs.get('window_size') is not None, 'Must provide a window size.'
        return SameDataIterator(dataset=dataset, pred_probs=pred_probs, num_reps=num_reps, fold=fold, window_size=int(kwargs['window_size']))
    else:
        raise ValueError('Unknown iterator for name: {}'.format(name))
