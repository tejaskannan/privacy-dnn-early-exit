import unittest
from privddnn.utils.ngrams import create_ngrams, create_ngram_counts


class NgramTests(unittest.TestCase):

    def test_2grams_2(self):
        levels = [0, 0, 1, 0, 1, 1, 0]
        preds = [9, 3, 2, 4, 1, 8, 3]

        ngram_inputs, ngram_outputs = create_ngrams(levels, preds, n=2, num_outputs=2)

        self.assertEqual([0, 1, 2, 1, 3, 2], ngram_inputs.tolist())

    def test_3grams_2(self):
        levels = [0, 0, 0, 1, 0, 1, 1, 0]
        preds = [0, 1, 1, 2, 2, 2, 0, 0]

        ngram_inputs, ngram_outputs = create_ngrams(levels, preds, n=3, num_outputs=2)

        self.assertEqual([0, 1, 2, 5, 3, 6], ngram_inputs.tolist())
        self.assertEqual([1, 1, 2, 2, 2, 0], ngram_outputs.tolist())

    def test_2grams_3(self):
        levels = [0, 2, 2, 1, 0, 2, 1, 0, 0]
        preds = [0, 1, 1, 2, 2, 2, 0, 0, 0]

        ngram_inputs, ngram_outputs = create_ngrams(levels, preds, n=2, num_outputs=3)

        self.assertEqual([2, 8, 7, 3, 2, 7, 3, 0], ngram_inputs.tolist())

    def test_3grams_3(self):
        levels = [0, 2, 2, 1, 0, 2, 1, 0, 0]
        preds = [0, 1, 1, 2, 2, 2, 0, 0, 0]

        ngram_inputs, ngram_outputs = create_ngrams(levels, preds, n=3, num_outputs=3)

        self.assertEqual([8, 25, 21, 11, 7, 21, 9], ngram_inputs.tolist())
        self.assertEqual([1, 1, 2, 2, 2, 0, 0], ngram_outputs.tolist())

    def test_counts_2gram_2(self):
        levels = [0, 0, 1, 0, 1, 1, 0]
        preds = [9, 3, 2, 4, 1, 8, 3]

        ngram_inputs, ngram_outputs = create_ngram_counts(levels, preds, n=2, num_outputs=2)
        self.assertEqual([6, 4, 4, 4, 2, 4], ngram_inputs.tolist())

    def test_counts_2gram_3(self):
        levels = [0, 0, 1, 0, 1, 1, 0]
        preds = [9, 3, 2, 4, 1, 8, 3]

        ngram_inputs, ngram_outputs = create_ngram_counts(levels, preds, n=2, num_outputs=2)
        self.assertEqual([6, 4, 4, 4, 2, 4], ngram_inputs.tolist())

    def test_counts_2gram_3(self):
        levels = [0, 0, 0, 1, 0, 1, 1, 1, 0]
        preds = [1, 1, 0, 0, 0, 2, 2, 2, 2]

        ngram_inputs, ngram_outputs = create_ngram_counts(levels, preds, n=3, num_outputs=2)
        self.assertEqual([12, 9, 9, 6, 6, 3, 6], ngram_inputs.tolist())
        self.assertEqual([1, 0, 0, 0, 2, 2, 2], ngram_outputs.tolist())

    def test_counts_3grams_2(self):
        levels = [0, 2, 2, 1, 0, 2, 1, 0, 0, 0]
        preds = [0, 1, 1, 2, 2, 2, 0, 0, 0, 0]

        ngram_inputs, ngram_outputs = create_ngram_counts(levels, preds, n=2, num_outputs=3)

        self.assertEqual([10, 2, 4, 12, 10, 4, 12, 18, 18], ngram_inputs.tolist())

    def test_counts_3grams_3(self):
        levels = [0, 2, 2, 1, 0, 2, 1, 0, 0, 0]
        preds = [0, 1, 1, 2, 2, 2, 0, 0, 0, 0]

        ngram_inputs, ngram_outputs = create_ngram_counts(levels, preds, n=3, num_outputs=3)

        self.assertEqual([18, 6, 21, 21, 21, 21, 36, 48], ngram_inputs.tolist())
        self.assertEqual([1, 1, 2, 2, 2, 0, 0, 0], ngram_outputs.tolist())


if __name__ == '__main__':
    unittest.main()
