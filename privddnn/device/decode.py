from collections import namedtuple
from enum import Enum, auto


EXIT_BYTE = 0x12
ELEVATE_BYTE = 0x34
BUFFERED_BYTE = 0x56

ElevateResult = namedtuple('ElevateResult', ['inputs', 'hidden'])


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
    inputs: List[float] = []
    for i in range(0, num_input_bytes, 2):
        idx = offset + i
        int_value = int.from_bytes(message[idx:idx+2], byteorder='big', signed=True)
        inputs.append(fixed_point_to_float(int_value, precision=precision))

    offset += num_input_bytes
    hidden: List[float] = []
    for i in range(0, num_hidden_bytes, 2):
        idx = offset + i
        int_value = int.from_bytes(message[idx:idx+2], byteorder='big', signed=True)
        hidden.append(fixed_point_to_float(int_value, precision=precision))

    return ElevateResult(inputs=inputs, hidden=hidden)
