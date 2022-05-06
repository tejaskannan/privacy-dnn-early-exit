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


def approx_softmax(array: np.ndarray, axis: int):
    """
    Normalizes the axis of the given array using an approximate
    softmax function. The approximation matches the piecewise linear function
    used to approximate exp(x) in fixed point arithmetic.
    """
    def linear_exp(x: np.ndarray):
        cond1 = (x >= (3.0 / 8.0)).astype(float)
        res1 = (2.0 + (2.0 * x) - (11.0 / 16.0)) * cond1

        cond2 = (x >= -(3.0 / 8.0)).astype(float) * (x < (3.0 / 8.0)).astype(float)
        res2 = (1.0 + x) * cond2

        cond3 = (x >= -1.0).astype(float) * (x < -(3.0 / 8.0)).astype(float)
        res3 = (0.5 + 0.5 * (x + (11.0 / 16.0))) * cond3

        cond4 = (x >= -1.75).astype(float) * (x < -1).astype(float)
        res4 = (0.25 + 0.25 * (x + 1.375)) * cond4

        cond5 = (x >= -2.75).astype(float) * (x < -1.75).astype(float)
        res5 = (0.125 + 0.125 * (x + (33.0 / 16.0))) * cond5

        cond6 = (x >= -3.5).astype(float) * (x < -2.75).astype(float)
        res6 = ((3.0 / 64.0) + (3.0 / 64.0) * (x + 3.0)) * cond6

        cond7 = (x >= -5).astype(float) * (x < -3.5).astype(float)
        res7 = ((1.0 / 64.0) + (1.0 / 64.0) * (x + 4.0)) * cond7

        return res1 + res2 + res3 + res4 + res5 + res6 + res7

    max_value = np.max(array, axis=axis, keepdims=True)
    exp_values = linear_exp(array - max_value)
    exp_sum = np.sum(exp_values, axis=axis, keepdims=True)

    return exp_values / exp_sum


