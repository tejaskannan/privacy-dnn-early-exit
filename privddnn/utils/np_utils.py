import numpy as np


def mask_non_max(array: np.ndarray) -> np.ndarray:
    """
    Sets all but the maximum (positive and negative absolute) value to zero.
    """
    assert len(array.shape) in (1, 2), 'Must provide a 1d or 2d array'

    original_shape = array.shape

    is_pos = (array > 0).astype(array.dtype)
    is_neg = (array < 0).astype(array.dtype)

    max_pos = np.max(array, axis=-1)
    max_neg = -1 * np.max(-1 * array, axis=-1)

    equals_pos = np.isclose(array, np.expand_dims(max_pos, axis=-1)).astype(array.dtype) * is_pos
    equals_neg = np.isclose(array, np.expand_dims(max_neg, axis=-1)).astype(array.dtype) * is_neg

    mask = equals_pos + equals_neg
    return mask * array
