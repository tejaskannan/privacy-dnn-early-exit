import tensorflow as tf2
import numpy as np
import hashlib
import os.path
from enum import Enum, auto
from typing import List, Tuple, Iterable

from privddnn.utils.loading import load_h5_dataset, load_npz_dataset
from privddnn.utils.constants import SMALL_NUMBER


MODULUS = 2**16
DATASET_NOISE = 0.25


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
        self._dataset_name = dataset_name
        dir_path = os.path.dirname(os.path.realpath(__file__))

        has_val_split = False
        self._is_noisy = False

        if dataset_name.endswith('_noisy'):
            dataset_name = dataset_name.replace('_noisy', '')
            self._is_noisy = True

        if dataset_name == 'mnist':
            tf_dataset = tf2.keras.datasets.mnist
            (X_train, y_train), (X_test, y_test) = tf_dataset.load_data()

            X_train = np.expand_dims(X_train, axis=-1)  # [N, 28, 28, 1]
            X_test = np.expand_dims(X_test, axis=-1)  # [M, 28, 28, 1]
        elif dataset_name == 'fashion_mnist':
            tf_dataset = tf2.keras.datasets.fashion_mnist
            (X_train, y_train), (X_test, y_test) = tf_dataset.load_data()

            X_train = np.expand_dims(X_train, axis=-1)  # [N, 28, 28, 1]
            X_test = np.expand_dims(X_test, axis=-1)  # [M, 28, 28, 1]
        elif dataset_name == 'cifar10':
            tf_dataset = tf2.keras.datasets.cifar10
            (X_train, y_train), (X_test, y_test) = tf_dataset.load_data()
        elif dataset_name == 'cifar100':
            dataset_name = 'cifar100'
            tf_dataset = tf2.keras.datasets.cifar100
            (X_train, y_train), (X_test, y_test) = tf_dataset.load_data()
        elif dataset_name == 'traffic_signs':
            X_train, y_train = load_h5_dataset(path=os.path.join('/local', 'traffic_signs', 'train.h5'))
            X_val, y_val = load_h5_dataset(path=os.path.join('/local', 'traffic_signs', 'val.h5'))
            X_test, y_test = load_h5_dataset(path=os.path.join('/local', 'traffic_signs', 'test.h5'))
            has_val_split = True
        elif dataset_name in ('uci_har', 'speech_commands', 'wisdm', 'emnist', 'wisdm_real', 'fashion_mnist_max_prob', 'fashion_mnist_label_max_prob'):
            X_train, y_train = load_h5_dataset(path=os.path.join(dir_path, '..', 'data', dataset_name, 'train.h5'))
            X_val, y_val = load_h5_dataset(path=os.path.join(dir_path, '..', 'data', dataset_name, 'val.h5'))
            X_test, y_test = load_h5_dataset(path=os.path.join(dir_path, '..', 'data', dataset_name, 'test.h5'))
            has_val_split = True
        else:
            raise ValueError('Unknown dataset with name: {}'.format(dataset_name))

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
    def num_features(self) -> int:
        return self.input_shape[-1]

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

    def fit_normalizer(self):
        if self._is_normalizer_fit:
            return

        input_shape = self._train_inputs.shape
        reshaped_inputs = self._train_inputs.reshape(-1, input_shape[-1])

        mean = np.average(reshaped_inputs, axis=0)
        std = np.std(reshaped_inputs, axis=0)

        ndims = len(input_shape)
        padding = (1, ) * (ndims - 1)
        self._mean = np.reshape(mean, padding + (input_shape[-1], ))
        self._std = np.reshape(std, padding + (input_shape[-1], ))

        self._is_normalizer_fit = True

    def normalize_data(self):
        if self.is_normalized:
            return

        assert self._is_normalizer_fit, 'Must call fit_normalizer() first'

        self._train_inputs = (self._train_inputs - self._mean) / (self._std + SMALL_NUMBER)
        self._val_inputs = (self._val_inputs - self._mean) / (self._std + SMALL_NUMBER)
        self._test_inputs = (self._test_inputs - self._mean) / (self._std + SMALL_NUMBER)

        # Apply noise after normalization (standardizes across datasets)
        if self._is_noisy:
            train_noise = self._rand.normal(loc=0.0, scale=DATASET_NOISE, size=self._train_inputs.shape)
            val_noise = self._rand.normal(loc=0.0, scale=DATASET_NOISE, size=self._val_inputs.shape)
            test_noise = self._rand.normal(loc=0.0, scale=DATASET_NOISE, size=self._test_inputs.shape)

            self._train_inputs += train_noise
            self._val_inputs += val_noise
            self._test_inputs += test_noise

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
