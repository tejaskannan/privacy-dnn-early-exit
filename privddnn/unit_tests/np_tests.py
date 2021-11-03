import unittest
import numpy as np
from privddnn.utils.np_utils import mask_non_max


class MaskTests(unittest.TestCase):

    def test_mask_1d_pos(self):
        array = np.array([1, 4, 2, 4])
        result = mask_non_max(array).tolist()

        self.assertEqual(result, [0, 4, 0, 4])

    def test_mask_1d_neg(self):
        array = np.array([-1, -4, -6, -2])
        result = mask_non_max(array).tolist()

        self.assertEqual(result, [0, 0, -6, 0])

    def test_mask_1d_mixed(self):
        array = np.array([1, -4, -6, -6, 10])
        result = mask_non_max(array).tolist()

        self.assertEqual(result, [0, 0, -6, -6, 10])

    def test_mask_2d_pos(self):
        array = np.array([[1, 4], [2, 1]])
        result = mask_non_max(array).tolist()

        self.assertEqual(result, [[0, 4], [2, 0]])

    def test_mask_2d_neg(self):
        array = np.array([[-1, -4, -6], [-8, -8, -3]])
        result = mask_non_max(array).tolist()

        self.assertEqual(result, [[0, 0, -6], [-8, -8, 0]])

    def test_mask_2d_mixed(self):
        array = np.array([[1, -4, -6], [2, 10, -4]])
        result = mask_non_max(array).tolist()

        self.assertEqual(result, [[1, 0, -6], [0, 10, -4]])


if __name__ == '__main__':
    unittest.main()
