import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List

from privddnn.utils.ngrams import create_ngrams
from privddnn.utils.metrics import get_joint_distribution
from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.plotting import PLOT_STYLE



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    args = parser.parse_args()

    val_log = read_json_gz(args.test_log)['val']['greedy_even']['0.5'][0]
    test_log = read_json_gz(args.test_log)['test']['greedy_even']['0.5'][0]
    n = 3

    val_levels = np.array(val_log['output_levels'])
    val_preds = np.array(val_log['preds'])

    test_levels = np.array(test_log['output_levels'])
    test_preds = np.array(test_log['preds'])

    # Compute the n-grams for each fold
    val_ngram_levels, val_ngram_preds = create_ngrams(levels=val_levels, preds=val_preds, n=n)
    test_ngram_levels, test_ngram_preds = create_ngrams(levels=test_levels, preds=test_preds, n=n)

    # Compute the joint distributions of each fold
    val_dist = get_joint_distribution(X=val_ngram_levels, Y=val_ngram_preds)
    test_dist = get_joint_distribution(X=test_ngram_levels, Y=test_ngram_preds)

    print(val_dist)
    print(test_dist)

    print('L1 Distance: {:.6f}'.format(np.sum(np.abs(val_dist - test_dist))))
