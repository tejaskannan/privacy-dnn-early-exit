import unittest
from typing import List

from privddnn.device.decode import decode_exit_message, get_message_type, MessageType, decode_elevate_message


class Decode(unittest.TestCase):

    def test_exit_message(self):
        message = b'\x12\x08'
        self.assertEqual(get_message_type(message), MessageType.EXIT)
        self.assertEqual(decode_exit_message(message), 8);

    def test_elevate_message(self):
        message = b'\x34\x04\x06\x07\x81\xfd\xb6\x00\x00\x08\x66\x08\x80'
        self.assertEqual(get_message_type(message), MessageType.ELEVATE)

        decoding_result = decode_elevate_message(message, precision=10)
        expected_inputs: List[float] = [1.8759765625, -0.572265625]
        expected_hidden: List[float] = [0, 2.099609375, 2.125]

        self.assertEqual(decoding_result.inputs, expected_inputs)
        self.assertEqual(decoding_result.hidden, expected_hidden)


if __name__ == '__main__':
    unittest.main()
