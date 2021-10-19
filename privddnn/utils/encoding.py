import numpy as np
import struct
from typing import List, Tuple


INT_OFFSET = 2**31
INT_SIZE = 4
INT_ORDER = 'big'
FLOAT_SIZE = 8


def encode_prediction(pred: int) -> bytes:
    return pred.to_bytes(INT_SIZE, INT_ORDER)


def decode_prediction(encoded: bytes) -> int:
    return int.from_bytes(encoded, INT_ORDER)


def encode(array: np.ndarray, dtype: str) -> bytes:
    assert dtype in ('float', 'int'), 'dtype must be either `float` or `int`. Got: {}'.format(dtype)

    data_encoder = encode_as_floats if (dtype == 'float') else encode_as_ints
    encoded_shape = encode_shape(array)
    encoded_values = data_encoder(array=array.reshape(-1))

    return encoded_shape + encoded_values


def decode(encoded: bytes, dtype: str) -> np.ndarray:
    assert dtype in ('float', 'int'), 'dtype must be either `float` or `int`. Got: {}'.format(dtype)

    data_decoder = decode_as_floats if (dtype == 'float') else decode_as_ints
    decoded_shape = decode_shape(encoded=encoded[0:INT_SIZE * 3])
    decoded_data = data_decoder(encoded=encoded[INT_SIZE * 3:])

    return np.reshape(decoded_data, newshape=decoded_shape)


def encode_shape(array: np.ndarray) -> bytes:
    assert (len(array.shape) <= 3) and (len(array.shape) >= 1), 'Array must be between 1d and 3d.'
    dim0 = array.shape[0]
    dim1 = array.shape[1] if len(array.shape) > 1 else 0
    dim2 = array.shape[2] if len(array.shape) > 2 else 0

    return dim0.to_bytes(INT_SIZE, INT_ORDER) + dim1.to_bytes(INT_SIZE, INT_ORDER) + dim2.to_bytes(INT_SIZE, INT_ORDER)


def decode_shape(encoded: bytes) -> Tuple[int, ...]:
    dim0 = int.from_bytes(encoded[0:INT_SIZE], INT_ORDER)
    dim1 = int.from_bytes(encoded[INT_SIZE:(2 * INT_SIZE)], INT_ORDER)
    dim2 = int.from_bytes(encoded[(2 * INT_SIZE):(3 * INT_SIZE)], INT_ORDER)

    if (dim1 == 0) and (dim2 == 0):
        return (dim0, )
    elif (dim2 == 0):
        return (dim0, dim1)
    else:
        return (dim0, dim1, dim2)


def encode_as_floats(array: np.ndarray) -> bytes:
    assert len(array.shape) == 1, 'Must flatten the array before encoding'
    
    encoded = bytearray()
    for value in array:
        enc = struct.pack('d', float(value))
        encoded.extend(enc)

    return bytes(encoded)


def decode_as_floats(encoded: bytes) -> List[float]:
    decoded: List[float] = []

    for idx in range(0, len(encoded), FLOAT_SIZE):
        start, end = idx, idx + FLOAT_SIZE
        enc = encoded[start:end]
        dec = struct.unpack('d', enc)
        decoded.append(dec)

    return decoded


def encode_as_ints(array: np.ndarray) -> bytes:
    assert len(array.shape) == 1, 'Must flatten the array before encoding'
     
    encoded = bytearray()
    for value in array:
        enc = (int(value) + INT_OFFSET).to_bytes(INT_SIZE, INT_ORDER)
        encoded.extend(enc)

    return bytes(encoded)


def decode_as_ints(encoded: bytes) -> List[int]:
    decoded: List[float] = []

    for idx in range(0, len(encoded), INT_SIZE):
        start, end = idx, idx + INT_SIZE
        enc = encoded[start:end]
        dec = int.from_bytes(enc, INT_ORDER) - INT_OFFSET
        decoded.append(dec)

    return decoded
