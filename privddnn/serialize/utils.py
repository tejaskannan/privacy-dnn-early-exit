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


def serialize_int_array(var_name: str, array: List[int], dtype: str) -> str:
    assert dtype in ('int16_t', 'int8_t', 'uint16_t', 'uint8_t'), 'Invalid data type: {}'.format(dtype)
    array_str = '{{ {} }}'.format(','.join(map(str, array)))
    return 'static {} {}[{}] = {};'.format(dtype, var_name, len(array), array_str)


def serialize_float_array(var_name: str, array: List[float], precision: int, width: int, dtype: str) -> str:
    assert dtype in ('int16_t', 'int8_t'), 'Invalid data type: {}'.format(dtype)
    array_str = '{{ {} }}'.format(','.join(map(lambda x: str(float_to_fixed_point(x, precision, width)), array)))
    return 'static {} {}[{}] = {};'.format(dtype, var_name, len(array), array_str)
