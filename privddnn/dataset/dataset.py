import tensorflow as tf2
import numpy as np
import hashlib
import os.path
from enum import Enum, auto
from typing import List, Tuple, Iterable

from privddnn.utils.loading import load_h5_dataset
from privddnn.utils.constants import SMALL_NUMBER


MODULUS = 2**16


class DataFold(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


def get_split_indices(num_samples: int, frac: float) -> Tuple[List[int], List[int]]:
    split_point = int(frac * MODULUS)
    one_indices: List[int] = []
    two_indices: List[int] = []

    for sample_idx in range(num_samples):
        h = hashlib.md5()
        h.update(sample_idx.to_bytes(length=8, byteorder='big'))
        digest = h.hexdigest()

        hash_result = int(digest, 16) % MODULUS

        if hash_result < split_point:
            one_indices.append(sample_idx)
        else:
            two_indices.append(sample_idx)

    return one_indices, two_indices


class Dataset:

    def __init__(self, dataset_name: str):
        # Get the dataset by name and load the data
        dataset_name = dataset_name.lower()
        dir_path = os.path.dirname(os.path.realpath(__file__))

        has_val_split = False

        if dataset_name == 'mnist':
            tf_dataset = tf2.keras.datasets.mnist
            (X_train, y_train), (X_test, y_test) = tf_dataset.load_data()
        elif dataset_name == 'fashion_mnist':
            tf_dataset = tf2.keras.datasets.fashion_mnist
            (X_train, y_train), (X_test, y_test) = tf_dataset.load_data()
        elif dataset_name == 'cifar_10':
            tf_dataset = tf2.keras.datasets.cifar10
            (X_train, y_train), (X_test, y_test) = tf_dataset.load_data()
        elif dataset_name == 'cifar_100':
            tf_dataset = tf2.keras.datasets.cifar100
            (X_train, y_train), (X_test, y_test) = tf_dataset.load_data()
        elif dataset_name == 'pen_digits':
            X_train, y_train = load_h5_dataset(path=os.path.join(dir_path, '..', 'data', 'pen_digits', 'train.h5'))
            X_test, y_test = load_h5_dataset(path=os.path.join(dir_path, '..', 'data', 'pen_digits', 'test.h5'))
        elif dataset_name == 'uci_har':
            X_train, y_train = load_h5_dataset(path=os.path.join(dir_path, '..', 'data', 'uci_har', 'train.h5'))
            X_val, y_val = load_h5_dataset(path=os.path.join(dir_path, '..', 'data', 'uci_har', 'val.h5'))
            X_test, y_test = load_h5_dataset(path=os.path.join(dir_path, '..', 'data', 'uci_har', 'test.h5'))
            has_val_split = True
        elif dataset_name == 'land_cover':
            X_train, y_train = load_h5_dataset(path=os.path.join(dir_path, '..', 'data', 'land_cover', 'train.h5'))
            X_val, y_val = load_h5_dataset(path=os.path.join(dir_path, '..', 'data', 'land_cover', 'val.h5'))
            X_test, y_test = load_h5_dataset(path=os.path.join(dir_path, '..', 'data', 'land_cover', 'test.h5'))
            has_val_split = True
        elif dataset_name == 'letter_recognition':
            X_train, y_train = load_h5_dataset(path=os.path.join(dir_path, '..', 'data', 'letter_recognition', 'train.h5'))
            X_val, y_val = load_h5_dataset(path=os.path.join(dir_path, '..', 'data', 'letter_recognition', 'val.h5'))
            X_test, y_test = load_h5_dataset(path=os.path.join(dir_path, '..', 'data', 'letter_recognition', 'test.h5'))
            has_val_split = True
        else:
            raise ValueError('Unknown dataset with name: {}'.format(dataset_name))

        self._dataset_name = dataset_name
        self._rand = np.random.RandomState(58924)

        # Make sure we have 1d label arrays
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        # Split the training set into a train and validation folds
        if has_val_split:
            y_val = y_val.reshape(-1)
        else:
            train_idx, val_idx = get_split_indices(num_samples=X_train.shape[0], frac=0.8)
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            X_train, y_train = X_train[train_idx], y_train[train_idx]

        # Organize the data folds
        self._train_inputs = X_train
        self._train_labels = y_train

        self._val_inputs = X_val
        self._val_labels = y_val

        self._test_inputs = X_test
        self._test_labels = y_test

        self._num_labels = np.amax(y_train) + 1

        # Track the state of the dataset
        self._is_normalized = False
        self._is_normalizer_fit = False

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def is_normalized(self) -> bool:
        return self._is_normalized

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._train_inputs.shape[1:]

    @property
    def num_labels(self) -> int:
        return self._num_labels

    @property
    def num_train(self) -> int:
        return self._train_inputs.shape[0]

    @property
    def num_val(self) -> int:
        return self._val_inputs.shape[0]

    @property
    def num_test(self) -> int:
        return self._test_inputs.shape[0]

    def get_inputs(self, fold: DataFold) -> np.ndarray:
        if fold == DataFold.TRAIN:
            return self.get_train_inputs()
        elif fold == DataFold.VAL:
            return self.get_val_inputs()
        elif fold == DataFold.TEST:
            return self.get_test_inputs()
        else:
            raise ValueError('Unknown data fold: {}'.format(fold.name))

    def get_labels(self, fold: DataFold) -> np.ndarray:
        if fold == DataFold.TRAIN:
            return self.get_train_labels()
        elif fold == DataFold.VAL:
            return self.get_val_labels()
        elif fold == DataFold.TEST:
            return self.get_test_labels()
        else:
            raise ValueError('Unknown data fold: {}'.format(fold.name))

    def get_train_inputs(self) -> np.ndarray:
        return self._train_inputs

    def get_train_labels(self) -> np.ndarray:
        return self._train_labels

    def get_val_inputs(self) -> np.ndarray:
        return self._val_inputs

    def get_val_labels(self) -> np.ndarray:
        return self._val_labels

    def get_test_inputs(self) -> np.ndarray:
        return self._test_inputs

    def get_test_labels(self) -> np.ndarray:
        return self._test_labels

    def fit_normalizer(self, is_global: bool):
        if self._is_normalizer_fit:
            return

        if is_global:
            self._mean = np.average(self._train_inputs)
            self._std = np.std(self._train_inputs)
        else:
            self._mean = np.expand_dims(np.average(self._train_inputs, axis=0), axis=0)
            self._std = np.expand_dims(np.std(self._train_inputs, axis=0), axis=0)

        self._is_normalizer_fit = True

    def normalize_data(self):
        if self.is_normalized:
            return

        assert self._is_normalizer_fit, 'Must call fit_normalizer() first'

        self._train_inputs = (self._train_inputs - self._mean) / (self._std + SMALL_NUMBER)
        self._val_inputs = (self._val_inputs - self._mean) / (self._std + SMALL_NUMBER)
        self._test_inputs = (self._test_inputs - self._mean) / (self._std + SMALL_NUMBER)

        self._is_normalized = True

    def generate_train_batches(self, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        return self.minibatch_generator(batch_size, fold=DataFold.TRAIN, should_shuffle=True)

    def generate_val_batches(self, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        return self.minibatch_generator(batch_size, fold=DataFold.VAL, should_shuffle=False)

    def generate_test_batches(self, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        return self.minibatch_generator(batch_size, fold=DataFold.TEST, should_shuffle=False)

    def minibatch_generator(self, batch_size: int, fold: DataFold, should_shuffle: bool) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        assert self.is_normalized, 'Must call normalize_data() first'

        # Get the inputs
        if fold == DataFold.TRAIN:
            inputs, labels = self._train_inputs, self._train_labels
        elif fold == DataFold.VAL:
            inputs, labels = self._val_inputs, self._val_labels
        elif fold == DataFold.TEST:
            inputs, labels = self._test_inputs, self._test_labels
        else:
            raise ValueError('Unknown data fold {}'.format(fold))

        # Generate the batches
        num_samples = inputs.shape[0]
        sample_idx = np.arange(num_samples)

        if should_shuffle:
            self._rand.shuffle(sample_idx)

        for idx in range(0, num_samples, batch_size):
            batch_inputs = inputs[idx:idx+batch_size]
            batch_labels = labels[idx:idx+batch_size]

            yield batch_inputs, batch_labels
