import numpy as np
from typing import List


def float_to_fixed_point(x: float, precision: int, width: int) -> int:
    mult = x * (1 << precision)
    max_value = (1 << (width - 1)) - 1

    if mult >= max_value:
        return max_value
    elif mult <= (-1 * max_value):
        return -1 * max_value
    else:
        return int(mult)


def array_to_fixed_point(x: np.ndarray, precision: int, width: int) -> np.ndarray:
    mult = x * (1 << precision)
    max_value = (1 << (width - 1)) - 1
    quantized = np.clip(mult, a_min=-max_value, a_max=max_value)
    return quantized.astype(int)


def serialize_int_array(var_name: str, array: List[int], dtype: str) -> str:
    assert dtype in ('int16_t', 'int8_t', 'uint16_t', 'uint8_t'), 'Invalid data type: {}'.format(dtype)
    array_str = '{{ {} }}'.format(','.join(map(str, array)))
    return 'static {} {}[{}] = {};'.format(dtype, var_name, len(array), array_str)


def serialize_float_array(var_name: str, array: List[float], precision: int, width: int, dtype: str) -> str:
    assert dtype in ('int16_t', 'int8_t'), 'Invalid data type: {}'.format(dtype)
    array_str = '{{ {} }}'.format(','.join(map(lambda x: str(float_to_fixed_point(x, precision, width)), array)))
    return 'static {} {}[{}] = {};'.format(dtype, var_name, len(array), array_str)


def expand_vector(vec: np.ndarray) -> np.array:
    """
    Expands the given vector to use 2 columns in preparation for the MSP430.
    The accelerator on the MSP430 requires an even number of dimensions for each matrix.
    """
    result = np.empty(2 * len(vec))

    for i in range(len(vec)):
        result[2 * i] = vec[i]
        result[2 * i + 1] = 0

    return result
