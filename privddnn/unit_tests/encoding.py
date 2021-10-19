import numpy as np
import unittest
from typing import List, Union

import privddnn.utils.encoding as encoding


class EncodeShape(unittest.TestCase):

    def test_encode_1d(self):
        array = np.random.uniform(size=(3, ))

        encoded_shape = encoding.encode_shape(array=array)
        decoded_shape = encoding.decode_shape(encoded=encoded_shape)

        self.assertEqual(encoded_shape, b'\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00')
        self.assertEqual(decoded_shape, array.shape)

    def test_encode_2d(self):
        array = np.random.uniform(size=(2, 3))

        encoded_shape = encoding.encode_shape(array=array)
        decoded_shape = encoding.decode_shape(encoded=encoded_shape)

        self.assertEqual(encoded_shape, b'\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x00')
        self.assertEqual(decoded_shape, array.shape)

    def test_encode_3d(self):
        array = np.random.uniform(size=(32, 28, 3))

        encoded_shape = encoding.encode_shape(array=array)
        decoded_shape = encoding.decode_shape(encoded=encoded_shape)

        self.assertEqual(encoded_shape, b'\x00\x00\x00\x20\x00\x00\x00\x1C\x00\x00\x00\x03')
        self.assertEqual(decoded_shape, array.shape)

    def test_encode_4d(self):
        array = np.random.uniform(size=(7, 3, 4, 5))

        with self.assertRaises(AssertionError) as context:
            encoding.encode_shape(array=array)

        self.assertTrue('Array must be between 1d and 3d.' == str(context.exception))


class EncodeData(unittest.TestCase):

    def compare_arrays(self, result: List[Union[float, int]], expected: np.ndarray):
        self.assertEqual(len(result), np.prod(expected.shape))
        for r, e in zip(result, expected.reshape(-1)):
            self.assertAlmostEqual(r, e)

    def test_encode_ints_3(self):
        array = np.array([1, 2, 3])
        encoded = encoding.encode_as_ints(array=array)
        decoded = encoding.decode_as_ints(encoded=encoded)

        self.compare_arrays(result=decoded, expected=array)

    def test_encode_ints_4(self):
        array = np.array([1, -2, 3, 5])
        encoded = encoding.encode_as_ints(array=array)
        decoded = encoding.decode_as_ints(encoded=encoded)

        self.compare_arrays(result=decoded, expected=array)

    def test_encode_ints_2d(self):
        array = np.array([[-1, 2], [4, 5], [12, -100]])
        encoded = encoding.encode_as_ints(array=array.reshape(-1))
        decoded = encoding.decode_as_ints(encoded=encoded)

        self.compare_arrays(result=decoded, expected=array)

    def test_encode_floats_3(self):
        array = np.array([1.01, 2.53, 3.123])
        encoded = encoding.encode_as_floats(array=array)
        decoded = encoding.decode_as_floats(encoded=encoded)

        self.compare_arrays(result=decoded, expected=array)

    def test_encode_floats_4(self):
        array = np.array([1.34, -2.23498, 3.1890, 5.134789])
        encoded = encoding.encode_as_floats(array=array)
        decoded = encoding.decode_as_floats(encoded=encoded)

        self.compare_arrays(result=decoded, expected=array)

    def test_encode_floats_2d(self):
        array = np.array([[-1.23489, 2.802934], [4.42389, 5.28390], [12.78931, -100.2384902]])
        encoded = encoding.encode_as_floats(array=array.reshape(-1))
        decoded = encoding.decode_as_floats(encoded=encoded)

        self.compare_arrays(result=decoded, expected=array)


if __name__ == '__main__':
    unittest.main()

