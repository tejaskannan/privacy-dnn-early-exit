import unittest
from privddnn.utils.exit_utils import get_exit_rates, normalize_exit_rates


class ExitUtilsTests(unittest.TestCase):

    def test_exit_rates_2(self):
        rates = get_exit_rates(single_rates=[0.4, 0.5, 0.6], num_outputs=2)
        expected = [[0.4, 0.6], [0.5, 0.5], [0.6, 0.4]]
        self.assertEqual(rates, expected)

    def test_exit_rates_3(self):
        rates = get_exit_rates(single_rates=[0.3, 0.4, 0.7], num_outputs=3)
        expected = [[0.3, 0.3, 0.4], [0.3, 0.4, 0.3], [0.3, 0.7, 0.0], [0.4, 0.3, 0.3], [0.4, 0.4, 0.2], [0.7, 0.3, 0.0]]
        self.assertEqual(rates, expected)


class NormalizeExitRatesTests(unittest.TestCase):

    def test_normalize_2(self):
        rates = [0.4, 0.6]
        normalized = normalize_exit_rates(rates=rates)
        expected = [0.4, 1.0]

        self.assertEqual(len(normalized), len(expected))
        
        for r, e in zip(normalized, expected):
            self.assertAlmostEqual(r, e)

    def test_normalize_2_zero(self):
        rates = [1.0, 0.0]
        normalized = normalize_exit_rates(rates=rates)
        expected = [1.0, 0.0]

        self.assertEqual(len(normalized), len(expected))
        
        for r, e in zip(normalized, expected):
            self.assertAlmostEqual(r, e)

    def test_normalize_3(self):
        rates = [0.4, 0.35, 0.25]
        normalized = normalize_exit_rates(rates=rates)
        expected = [0.4, 0.58333333333, 1.0]

        self.assertEqual(len(normalized), len(expected))
        
        for r, e in zip(normalized, expected):
            self.assertAlmostEqual(r, e)

    def test_normalize_4(self):
        rates = [0.1, 0.5, 0.3, 0.1]
        normalized = normalize_exit_rates(rates=rates)
        expected = [0.1, 0.5555555555, 0.75, 1.0]

        self.assertEqual(len(normalized), len(expected))
        
        for r, e in zip(normalized, expected):
            self.assertAlmostEqual(r, e)


if __name__ == '__main__':
    unittest.main()


