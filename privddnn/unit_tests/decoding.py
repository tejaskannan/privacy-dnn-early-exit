import unittest
from typing import List

from privddnn.device.decode import decode_exit_message, get_message_type, MessageType, decode_elevate_message
from privddnn.device.decode import decode_exit_bitmask, decode_predictions, decode_buffered_message


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

    def test_bitmask(self):
        message = b'\x56\x20\x30\x0a\x1b\x03\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08'
        self.assertEqual(get_message_type(message), MessageType.BUFFERED)

        did_elevate, offset = decode_exit_bitmask(message, 4, 10)

        expected = [0, 1, 3, 4, 8, 9]
        self.assertEqual(did_elevate, expected)
        self.assertEqual(offset, 6)

    def test_predictions(self):
        message = b'\x56\x20\x30\x0a\x1b\x03\x08\x08\x08\x04\x08\x07\x08\x08\x08\x08'
        self.assertEqual(get_message_type(message), MessageType.BUFFERED)

        preds, offset = decode_predictions(message, 6, 10)

        expected = [8, 8, 8, 4, 8, 7, 8, 8, 8, 8]
        self.assertEqual(preds, expected)
        self.assertEqual(offset, 16)

    def test_buffered_message(self):
        message = b'\x56\x04\x06\x02\x01\x08\x04\x07\x81\xfd\xb6\x00\x00\x08\x66\x08\x80'
        self.assertEqual(get_message_type(message), MessageType.BUFFERED)

        buffered_result = decode_buffered_message(message, precision=10)

        expected_preds = [8, 4]
        self.assertEqual(buffered_result.preds, expected_preds)

        expected_indices = [0]
        self.assertEqual(buffered_result.elevate_indices, expected_indices)

        expected_inputs: List[float] = [1.8759765625, -0.572265625]
        expected_hidden: List[float] = [0, 2.099609375, 2.125]

        self.assertEqual(buffered_result.inputs[0], expected_inputs)
        self.assertEqual(buffered_result.hidden[0], expected_hidden)

if __name__ == '__main__':
    unittest.main()
