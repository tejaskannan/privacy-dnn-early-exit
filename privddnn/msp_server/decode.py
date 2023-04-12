import numpy as np
from enum import Enum, auto
from collections import namedtuple
from encryption import decrypt_aes128


Response = namedtuple('Response', ['response_type', 'value'])
MAX_VALUE = 32767


class ResponseType(Enum):
    EXIT = auto()
    CONTINUE = auto()



def to_fixed_point(value: float, precision: int) -> int:
    factor = (1 << precision)
    casted = int(value * factor)

    if casted > MAX_VALUE:
        return MAX_VALUE
    elif casted < -MAX_VALUE:
        return -MAX_VALUE
    else:
        return casted


def from_fixed_point(value: int, precision: int) -> int:
    return float(value) / float(1 << precision)


def encode_inputs(inputs: np.ndarray, precision: int) -> bytes:
    # Flatten
    inputs = inputs.reshape(-1)

    # Convert to fixed point
    fp_values = [to_fixed_point(val, precision) for val in inputs]

    print(fp_values)
    
    # Write into an array of bytes
    encoded_values = [val.to_bytes(2, byteorder='big', signed=True) for val in fp_values]
    
    result = bytes()
    for encoded in encoded_values:
        result += encoded

    return result


def decode_response(message: bytes, key: bytes, precision: int) -> Response:
    print(message)

    # Decrypt the message
    plaintext = decrypt_aes128(ciphertext=message, key=key)

    print(plaintext)

    # Return the prediction (first byte of the result)
    control_byte = plaintext[0]
    length = int.from_bytes(plaintext[1:3], byteorder='big')
    offset = 3

    response_type = ResponseType.EXIT if control_byte == 0 else ResponseType.CONTINUE

    if response_type == ResponseType.CONTINUE:
        value_bytes = plaintext[offset:offset+length]
        value_list: List[float] = []

        for idx in range(0, len(value_bytes), 2):
            value = int.from_bytes(value_bytes[idx:idx+offset], byteorder='big')
            value_list.append(from_fixed_point(value, precision))

        value = np.array(value_list)
    elif response_type == ResponseType.EXIT:
        value = plaintext[offset]
    else:
        raise ValueError('Invalid response type: {}'.format(response_type))

    return Response(response_type=response_type, value=value)
