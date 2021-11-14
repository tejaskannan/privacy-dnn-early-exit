import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List

from privddnn.utils.ngrams import create_ngrams, create_ngram_counts
from privddnn.utils.metrics import compute_mutual_info
from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.plotting import PLOT_STYLE


WIDTH = 0.08


def partition(ngrams: np.ndarray, preds: np.ndarray, num_ngrams: int) -> List[np.ndarray]:
    result: List[np.ndarray] = []
    num_labels = np.max(preds) + 1

    for label in range(num_labels):

        ngram_counts = np.zeros(shape=(num_ngrams, ))
        for ngram, pred in zip(ngrams, preds):
            if label == pred:
                ngram_counts[ngram] += 1

        result.append(ngram_counts)

    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    args = parser.parse_args()

    policy_name = 'random'
    rate = '0.5'
    test_log = read_json_gz(args.test_log)['val'][policy_name][rate][0]
    n = 7

    output_levels = np.array(test_log['output_levels'])
    preds = np.array(test_log['preds'])

    #sample_idx = np.arange(len(preds))
    #np.random.shuffle(sample_idx)

    #output_levels = output_levels[sample_idx]
    #preds = preds[sample_idx]

    ngram_levels, ngram_preds = create_ngrams(levels=output_levels, preds=preds, n=n)
    count_levels, count_preds = create_ngram_counts(levels=output_levels, preds=preds, n=n)

    ngram_mut_info = compute_mutual_info(ngram_levels, ngram_preds, should_normalize=False)
    count_mut_info = compute_mutual_info(count_levels, count_preds, should_normalize=False)
    standard_mut_info = compute_mutual_info(output_levels, preds, should_normalize=False)

    print('Policy: {}, Rate: {}'.format(policy_name, rate))
    print('{}-Gram Mutual Information: {}'.format(n, ngram_mut_info))
    print('Count Mutual Information: {}'.format(count_mut_info))
    print('Standard Mutual Information: {}'.format(standard_mut_info))

#    num_ngrams = (1 << n)
#    ngram_partitions = partition(ngrams=ngram_levels, preds=ngram_preds, num_ngrams=num_ngrams)
#
#    print(np.bincount(ngram_preds))
#
#    with plt.style.context(PLOT_STYLE):
#        fig, ax = plt.subplots()
#
#        num_labels = len(ngram_partitions)
#        xs = np.arange(num_ngrams)
#        offset = (-(num_labels - 1) / 2) * WIDTH
#
#        for i in range(num_labels):
#            ax.bar(xs - offset, ngram_partitions[i], width=WIDTH)
#            offset += WIDTH
#
#        #ax.bar(xs, ngram_counts, width=WIDTH)
#
#        plt.show()
#
