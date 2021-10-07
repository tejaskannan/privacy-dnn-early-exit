import unittest
import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np

from privddnn.utils.tf_utils import make_max_prob_targets


class MaxProbTests(unittest.TestCase):

    def test_three_labels(self):
        num_labels = 3
        target_list = [0.6, 0.4]
        label_list = [1, 2, 0, 2]

        with tf1.Session(graph=tf2.Graph()) as sess:
            labels = tf2.constant(label_list)
            target = tf2.constant(target_list)
            target_dist = make_max_prob_targets(labels=labels, num_labels=num_labels, target_prob=target)
            result = sess.run(target_dist)

            self.assertEqual(result.shape, (4, 2, 3))

            for idx, label in enumerate(label_list):
                for level in range(result.shape[1]):
                    self.assertAlmostEqual(np.sum(result[idx, level], axis=-1), 1.0)
                    self.assertEqual(np.argmax(result[idx, level]), label)
                    self.assertAlmostEqual(np.max(result[idx, level]), target_list[level])


if __name__ == '__main__':
    unittest.main()
