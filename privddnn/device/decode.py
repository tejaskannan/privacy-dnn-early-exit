import math
from collections import namedtuple
from enum import Enum, auto
from typing import Tuple, List


EXIT_BYTE = 0x12
ELEVATE_BYTE = 0x34
BUFFERED_BYTE = 0x56
BITS_PER_BYTE = 8

ElevateResult = namedtuple('ElevateResult', ['inputs', 'hidden'])
BufferedResult = namedtuple('BufferedResult', ['inputs', 'hidden', 'elevate_indices', 'preds'])


class MessageType:
    EXIT = auto()
    ELEVATE = auto()
    BUFFERED = auto()


def fixed_point_to_float(x: int, precision: int) -> float:
    return float(x) / float(1 << precision)


def get_message_type(message: bytes) -> MessageType:
    control_byte = message[0]

    if (control_byte == EXIT_BYTE):
        return MessageType.EXIT
    elif (control_byte == ELEVATE_BYTE):
        return MessageType.ELEVATE
    elif (control_byte == BUFFERED_BYTE):
        return MessageType.BUFFERED
    else:
        raise ValueError('Unknown control byte: {}'.format(control_byte))


def decode_exit_message(message: bytes) -> int:
    return message[1]


def decode_elevate_message(message: bytes, precision: int) -> ElevateResult:
    # Get the number of bytes in both the input and hidden states
    num_input_bytes = message[1]
    num_hidden_bytes = message[2]

    offset = 3
    inputs, offset = decode_vector(message=message,
                                   offset=offset,
                                   num_bytes=num_input_bytes,
                                   precision=precision)

    hidden, offset = decode_vector(message=message,
                                   offset=offset,
                                   num_bytes=num_hidden_bytes,
                                   precision=precision)

    return ElevateResult(inputs=inputs, hidden=hidden)


def decode_buffered_message(message: bytes, precision: int) -> BufferedResult:
    # Get the number of bytes in the input and hidden states, as well as the window size
    num_input_bytes = message[1]
    num_hidden_bytes = message[2]
    window_size = message[3]

    # Decode the exit bitmask to get the indices of the samples which should be elevated
    offset = 4
    elevate_indices, offset = decode_exit_bitmask(message, offset, window_size)
    num_elevated = len(elevate_indices)

    # Decode the predictions
    preds, offset = decode_predictions(message, offset, window_size)

    # Decode the inputs and hidden states
    inputs: List[List[float]] = []
    hidden: List[List[float]] = []

    for i in range(num_elevated):
        input_vector, offset = decode_vector(message=message,
                                             offset=offset,
                                             num_bytes=num_input_bytes,
                                             precision=precision)

        hidden_vector, offset = decode_vector(message=message,
                                              offset=offset,
                                              num_bytes=num_hidden_bytes,
                                              precision=precision)

        inputs.append(input_vector)
        hidden.append(hidden_vector)

    return BufferedResult(inputs=inputs, hidden=hidden, elevate_indices=elevate_indices, preds=preds)


def decode_vector(message: bytes, offset: int, num_bytes: int, precision: int) -> Tuple[List[float], int]:
    result: List[float] = []

    for i in range(offset, offset + num_bytes, 2):
        int_value = int.from_bytes(message[i:i+2], byteorder='big', signed=True)
        result.append(fixed_point_to_float(int_value, precision=precision))

    return result, offset + num_bytes


def decode_exit_bitmask(message: bytes, offset: int, window_size: int) -> Tuple[List[int], int]:
    """
    Decodes the exit bitmask and returns the indices of the samples
    that got elevated in the window.
    """
    num_bytes = int(math.ceil(window_size / BITS_PER_BYTE))
    result: List[int] = []
    idx = 0

    for i in range(offset, offset + num_bytes):
        for j in range(BITS_PER_BYTE):
            elevate_bit = (message[i] >> j) & 1

            if elevate_bit == 1:
                result.append(idx)

            idx += 1

    return result, offset + num_bytes


def decode_predictions(message: bytes, offset: int, window_size: int) -> Tuple[List[int], int]:
    """
    Decodes the predictions from each inference in the window.
    """
    return [message[i] for i in range(offset, offset + window_size)], offset + window_size
