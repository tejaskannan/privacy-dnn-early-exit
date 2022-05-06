import numpy as np
from typing import List, Union


def float_to_fixed_point(x: float, precision: int, width: int) -> int:
    mult = x * (1 << precision)
    max_value = (1 << (width - 1)) - 1

    if mult >= max_value:
        return max_value
    elif mult <= (-1 * max_value):
        return -1 * max_value
    else:
        return int(mult)


def array_to_fixed_point(x: Union[np.ndarray, List[float]], precision: int, width: int) -> np.ndarray:
    if isinstance(x, list):
        x = np.array(x)

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


def serialize_block_matrix(var_name: str, matrix: np.ndarray, block_size: int, precision: int, width: int, dtype: str, is_msp: bool) -> str:
    assert dtype in ('int16_t', 'int8_t'), 'Invalid data type: {}'.format(dtype)

    var_list: List[str] = []

    num_blocks = 0
    rows: List[int] = []
    cols: List[int] = []
    block_mat_names: List[str] = []

    for row in range(0, matrix.shape[0], block_size):
        for col in range(0, matrix.shape[1], block_size):
            block = matrix[row:row+block_size, col:col+block_size]

            assert (block.shape[0] % 2 == 0) and (block.shape[1] % 2 == 0), 'Block dimensions must be even. Got {}'.format(block.shape)

            block_data_name = '{}_BLOCK_DATA_{}'.format(var_name, num_blocks)
            block_data = serialize_float_array(var_name=block_data_name,
                                               array=block.reshape(-1).astype(float).tolist(),
                                               precision=precision,
                                               width=width,
                                               dtype=dtype)

            if is_msp:
                var_list.append('#pragma PERSISTENT({})'.format(block_data_name))

            var_list.append(block_data)

            block_mat_name = '{}_BLOCK_{}'.format(var_name, num_blocks)
            block_mat = 'static struct matrix {} = {{ {}, {}, {} }};'.format(block_mat_name, block_data_name, block.shape[0], block.shape[1])
            var_list.append(block_mat)

            block_mat_names.append('&{}'.format(block_mat_name))
            rows.append(row)
            cols.append(col)
            num_blocks += 1

    blocks_name = '{}_BLOCKS'.format(var_name)
    blocks_var = 'static struct matrix *{}[] = {{ {} }};'.format(blocks_name, ','.join(block_mat_names))
    var_list.append(blocks_var)

    rows_name = '{}_ROWS'.format(var_name)
    rows_var = 'static uint8_t {}[] = {{ {} }};'.format(rows_name, ','.join(map(str, rows)))
    var_list.append(rows_var)

    cols_name = '{}_COLS'.format(var_name)
    cols_var = 'static uint8_t {}[] = {{ {} }};'.format(cols_name, ','.join(map(str, cols)))
    var_list.append(cols_var)

    mat_var = 'static struct block_matrix {} = {{ {}, {}, {}, {}, {}, {} }};'.format(var_name, blocks_name, num_blocks, matrix.shape[0], matrix.shape[1], rows_name, cols_name)
    var_list.append(mat_var)

    return '\n'.join(var_list)


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
