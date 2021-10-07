import unittest
import numpy as np

from privddnn.utils.metrics import compute_max_prob_metric, compute_entropy_metric


class MaxProbTests(unittest.TestCase):

    def test_three_labels(self):
        probs = np.array([[0.1, 0.9, 0.0], [0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.01, 0.04, 0.95]])
        metrics = compute_max_prob_metric(probs=probs)
        expected = [0.9, 0.7, 0.6, 0.95]

        self.assertEqual(len(metrics), len(expected))

        for m, e in zip(metrics, expected):
            self.assertAlmostEqual(m, e)


class EntropyTests(unittest.TestCase):

    def test_three_labels(self):
        probs = np.array([[0.1, 0.9, 0.0], [0.7, 0.2, 0.1]])
        metrics = compute_entropy_metric(probs=probs)
        expected = [0.704096726723065559683, 0.270153301575614942003835517]

        self.assertEqual(len(metrics), len(expected))

        for m, e in zip(metrics, expected):
            self.assertAlmostEqual(m, e)

    def test_three_labels_bounds(self):
        probs = np.array([[0.0, 1.0, 0.0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
        metrics = compute_entropy_metric(probs=probs)
        expected = [1.0, 0.0]

        self.assertEqual(len(metrics), len(expected))

        for m, e in zip(metrics, expected):
            self.assertAlmostEqual(m, e)



if __name__ == '__main__':
    unittest.main()
