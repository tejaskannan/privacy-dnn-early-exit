import unittest
import numpy as np
from privddnn.utils.metrics import compute_entropy, compute_conditional_entropy, get_joint_distribution, compute_joint_entropy


class EntropyTests(unittest.TestCase):

    def test_entropy_2_even(self):
        entropy = compute_entropy(np.array([0.5, 0.5]), axis=0)
        self.assertAlmostEqual(entropy, 1.0, places=7)

    def test_entropy_2_uneven(self):
        entropy = compute_entropy(np.array([0.75, 0.25]), axis=0)
        self.assertAlmostEqual(entropy, 0.811278124)

    def test_entropy_2_zero(self):
        entropy = compute_entropy(np.array([1.0, 0.0]), axis=0)
        self.assertAlmostEqual(entropy, 0.0, places=7)

    def test_entropy_4_even(self):
        entropy = compute_entropy(np.array([0.25, 0.25, 0.25, 0.25]), axis=0)
        self.assertAlmostEqual(entropy, 2.0, places=7)

    def test_entropy_4_uneven(self):
        entropy = compute_entropy(np.array([0.4, 0.0, 0.21, 0.39]), axis=0)
        self.assertAlmostEqual(entropy, 1.531391428, places=7)

    def test_entropy_24_last_axis(self):
        entropy = compute_entropy(np.array([[0.4, 0.0, 0.21, 0.39], [0.25, 0.25, 0.25, 0.25]]), axis=-1)

        self.assertEqual(entropy.shape, (2, ))
        self.assertAlmostEqual(entropy[0], 1.531391428, places=7)
        self.assertAlmostEqual(entropy[1], 2.0, places=7)

    def test_entropy_24_first_axis(self):
        entropy = compute_entropy(np.array([[0.5, 0.75, 1.0, 0.1], [0.5, 0.25, 0.0, 0.9]]), axis=0)

        self.assertEqual(entropy.shape, (4, ))
        self.assertAlmostEqual(entropy[0], 1.0, places=7)
        self.assertAlmostEqual(entropy[1], 0.811278124, places=7)
        self.assertAlmostEqual(entropy[2], 0.0, places=7)
        self.assertAlmostEqual(entropy[3], 0.468995594, places=7)


class ConditionalEntropyTests(unittest.TestCase):

    def test_cond_entropy_2_2_even(self):
        joint_probs = np.array([[0.25, 0.25], [0.25, 0.25]])
        cond_entropy = compute_conditional_entropy(joint_probs)
        self.assertAlmostEqual(cond_entropy, 1.0, places=7)

    def test_cond_entropy_3_3_uneven(self):
        joint_probs = np.array([[0.1, 0.0, 0.0], [0.2, 0.3, 0.2], [0.0, 0.0, 0.2]])
        cond_entropy = compute_conditional_entropy(joint_probs)
        self.assertAlmostEqual(cond_entropy, 0.6754887502, places=7)

    def test_cond_entropy_3_3_uneven_2(self):
        joint_probs = np.array([[0.1, 0.0, 0.0], [0.2, 0.3, 0.2], [0.0, 0.0, 0.2]])
        cond_entropy = compute_conditional_entropy(joint_probs.T)
        self.assertAlmostEqual(cond_entropy, 1.0896596952, places=7)

    def test_cond_entropy_2_3_uneven(self):
        joint_probs = np.array([[0.0, 0.0], [0.1, 0.3], [0.4, 0.2]])
        cond_entropy = compute_conditional_entropy(joint_probs)
        self.assertAlmostEqual(cond_entropy, 0.8464393447, places=7)


class JointEntropyTests(unittest.TestCase):

    def test_joint_entropy_2_2_even(self):
        joint_probs = np.array([[0.25, 0.25], [0.25, 0.25]])
        joint_entropy = compute_joint_entropy(joint_probs)
        self.assertAlmostEqual(joint_entropy, 2.0, places=7)

    def test_joint_entropy_3_3_uneven(self):
        joint_probs = np.array([[0.1, 0.0, 0.0], [0.2, 0.3, 0.2], [0.0, 0.0, 0.2]])
        joint_entropy = compute_joint_entropy(joint_probs)
        self.assertAlmostEqual(joint_entropy, 2.24643934467, places=7)

    def test_cond_entropy_2_3_uneven(self):
        joint_probs = np.array([[0.0, 0.0], [0.1, 0.3], [0.4, 0.2]])
        joint_entropy = compute_joint_entropy(joint_probs)
        self.assertAlmostEqual(joint_entropy, 1.84643934467, places=7)


class JointDistributionTests(unittest.TestCase):

    def test_joint_distribution_2_2_even(self):
        X = np.array([0, 1, 0, 1])
        Y = np.array([1, 1, 0, 0])

        joint_probs = get_joint_distribution(X=X, Y=Y)

        self.assertEqual(joint_probs.shape, (2, 2))
        self.assertAlmostEqual(joint_probs[0, 0], 0.25, places=7)
        self.assertAlmostEqual(joint_probs[0, 1], 0.25, places=7)
        self.assertAlmostEqual(joint_probs[1, 0], 0.25, places=7)
        self.assertAlmostEqual(joint_probs[1, 1], 0.25, places=7)

    def test_joint_distribution_2_2_uneven(self):
        X = np.array([0, 1, 0, 1])
        Y = np.array([1, 1, 1, 0])

        joint_probs = get_joint_distribution(X=X, Y=Y)

        self.assertEqual(joint_probs.shape, (2, 2))
        self.assertAlmostEqual(joint_probs[0, 0], 0.00, places=7)
        self.assertAlmostEqual(joint_probs[0, 1], 0.50, places=7)
        self.assertAlmostEqual(joint_probs[1, 0], 0.25, places=7)
        self.assertAlmostEqual(joint_probs[1, 1], 0.25, places=7)

    def test_joint_distribution_2_3_uneven(self):
        X = np.array([0, 1, 0, 1, 0, 0, 1, 0])
        Y = np.array([1, 1, 1, 0, 2, 2, 2, 2])

        joint_probs = get_joint_distribution(X=X, Y=Y)

        self.assertEqual(joint_probs.shape, (2, 3))
        self.assertAlmostEqual(joint_probs[0, 0], 0.00, places=7)
        self.assertAlmostEqual(joint_probs[0, 1], 0.25, places=7)
        self.assertAlmostEqual(joint_probs[1, 0], 0.125, places=7)
        self.assertAlmostEqual(joint_probs[1, 1], 0.125, places=7)
        self.assertAlmostEqual(joint_probs[0, 2], 0.375, places=7)
        self.assertAlmostEqual(joint_probs[1, 2], 0.125, places=7)


if __name__ == '__main__':
    unittest.main()
