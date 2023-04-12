import unittest
from privddnn.utils.metrics import compute_mutual_info
from privddnn.utils.ngrams import create_ngrams, create_ngram_counts


class NgramTests(unittest.TestCase):

    def test_2grams_2(self):
        levels = [0, 0, 1, 0, 1, 1, 0]
        preds = [9, 3, 2, 4, 1, 8, 3]

        ngram_inputs, ngram_outputs = create_ngrams(levels, preds, n=2, num_outputs=2, num_clusters=3)

        self.assertEqual([2, 0, 1], ngram_inputs.tolist())
        self.assertEqual([9, 2, 1], ngram_outputs.tolist())

    def test_3grams_2(self):
        levels = [0, 0, 0, 1, 0, 1, 1, 0]
        preds = [0, 1, 1, 2, 2, 2, 0, 0]

        ngram_inputs, ngram_outputs = create_ngrams(levels, preds, n=3, num_outputs=2, num_clusters=7)

        self.assertEqual([0, 5], ngram_inputs.tolist())
        self.assertEqual([1, 2], ngram_outputs.tolist())

    def test_3grams_2_v2(self):
        levels = [0, 0, 0, 1, 0, 1, 1, 0]
        preds = [0, 1, 1, 2, 2, 2, 0, 0]

        ngram_inputs, ngram_outputs = create_ngrams(levels, preds, n=3, num_outputs=2, num_clusters=1)

        self.assertEqual([0, 0], ngram_inputs.tolist())
        self.assertEqual([1, 2], ngram_outputs.tolist())

    def test_2grams_3(self):
        levels = [0, 2, 2, 1, 0, 2, 1, 0, 0]
        preds = [0, 1, 1, 2, 2, 2, 0, 0, 0]

        ngram_inputs, ngram_outputs = create_ngrams(levels, preds, n=2, num_outputs=3, num_clusters=3)

        self.assertEqual([1, 0, 1, 2], ngram_inputs.tolist())
        self.assertEqual([0, 1, 2, 0], ngram_outputs.tolist())

    def test_3grams_3(self):
        levels = [0, 2, 2, 1, 0, 2, 1, 0, 0]
        preds = [0, 1, 1, 2, 2, 2, 0, 0, 0]

        ngram_inputs, ngram_outputs = create_ngrams(levels, preds, n=3, num_outputs=3, num_clusters=3)

        self.assertEqual([1, 0, 2], ngram_inputs.tolist())
        self.assertEqual([1, 2, 0], ngram_outputs.tolist())

    def test_2grams_3_mutual_info(self):
        levels = [0, 2, 2, 1, 0, 2, 1, 0, 0]
        preds = [0, 1, 1, 2, 2, 2, 0, 0, 0]

        ngram_inputs, ngram_outputs = create_ngrams(levels, preds, n=2, num_outputs=3, num_clusters=3)
        mutual_information = compute_mutual_info(X=ngram_inputs, Y=ngram_outputs, should_normalize=True)

        preds_entropy = 1.5
        window_entropy = 1.5
        joint_entropy = 2.0

        expected = (2 * (preds_entropy + window_entropy - joint_entropy)) / (preds_entropy + window_entropy)
        self.assertAlmostEqual(mutual_information, expected, places=7)

    def test_counts_2gram_2(self):
        levels = [0, 0, 1, 0, 1, 1, 0, 1]
        preds = [9, 3, 2, 4, 1, 8, 3, 3]

        ngram_inputs, ngram_outputs = create_ngram_counts(levels, preds, n=2, num_outputs=2, num_clusters=3)
        self.assertEqual([0, 1, 2, 1], ngram_inputs.tolist())

    def test_counts_2gram_3(self):
        levels = [0, 0, 0, 1, 0, 1, 1, 1, 0]
        preds = [1, 1, 0, 0, 0, 2, 2, 2, 2]

        ngram_inputs, ngram_outputs = create_ngram_counts(levels, preds, n=3, num_outputs=2, num_clusters=3)
        self.assertEqual([12, 6, 6], ngram_inputs.tolist())
        self.assertEqual([1, 0, 2], ngram_outputs.tolist())

    def test_counts_2gram_3(self):
        levels = [0, 0, 0, 1, 0, 1, 1, 1, 0]
        preds = [1, 1, 0, 0, 0, 2, 2, 2, 2]

        ngram_inputs, ngram_outputs = create_ngram_counts(levels, preds, n=3, num_outputs=2, num_clusters=2)
        self.assertEqual([0, 1, 1], ngram_inputs.tolist())
        self.assertEqual([1, 0, 2], ngram_outputs.tolist())

    def test_counts_3grams_2(self):
        levels = [0, 2, 2, 1, 0, 2, 1, 0, 0, 0]
        preds = [0, 1, 1, 2, 2, 2, 0, 0, 0, 0]

        ngram_inputs, ngram_outputs = create_ngram_counts(levels, preds, n=2, num_outputs=3, num_clusters=3)

        self.assertEqual([1, 2, 1, 0, 0], ngram_inputs.tolist())
        self.assertEqual([0, 1, 2, 0, 0], ngram_outputs.tolist())

    def test_counts_3grams_3(self):
        levels = [0, 2, 2, 1, 0, 2, 1, 0, 0, 0]
        preds = [0, 1, 1, 2, 2, 2, 0, 0, 0, 0]

        ngram_inputs, ngram_outputs = create_ngram_counts(levels, preds, n=3, num_outputs=3, num_clusters=3)

        self.assertEqual([0, 2, 1], ngram_inputs.tolist())
        self.assertEqual([1, 2, 0], ngram_outputs.tolist())

    def test_counts_3grams_2_mutual_info(self):
        levels = [0, 2, 2, 1, 0, 2, 1, 0, 0, 0]
        preds = [0, 1, 1, 2, 2, 2, 0, 0, 0, 0]

        ngram_inputs, ngram_outputs = create_ngram_counts(levels, preds, n=2, num_outputs=3, num_clusters=4)

        mutual_information = compute_mutual_info(X=ngram_inputs, Y=ngram_outputs, should_normalize=True)

        preds_entropy = 1.3709505945
        window_entropy = 1.9219280949
        joint_entropy = 2.3219280949
        expected = (2 * (preds_entropy + window_entropy - joint_entropy)) / (preds_entropy + window_entropy)
        self.assertAlmostEqual(mutual_information, expected, places=7)


if __name__ == '__main__':
    unittest.main()
