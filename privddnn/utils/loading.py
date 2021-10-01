import h5py
import numpy as np
from typing import Tuple


def load_h5_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, 'r') as fin:
        inputs = fin['inputs'][:]
        labels = fin['labels'][:]

    return inputs, labels
