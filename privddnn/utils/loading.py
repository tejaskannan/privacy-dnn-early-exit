import h5py
import os.path
import numpy as np
from typing import Tuple


def load_h5_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, 'r') as fin:
        inputs = fin['inputs'][:]
        labels = fin['labels'][:]

    return inputs, labels


def load_npz_dataset(path: str, fold: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(os.path.join(path, '{}-inputs.npz'.format(fold))) as data:
        inputs = data['arr_0']

    with np.load(os.path.join(path, '{}-labels.npz'.format(fold))) as data:
        labels = data['arr_0']

    return inputs, labels
