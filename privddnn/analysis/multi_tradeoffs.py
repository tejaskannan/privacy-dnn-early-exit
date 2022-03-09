import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List

from privddnn.dataset.dataset import Dataset
from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info, compute_geometric_mean
from privddnn.utils.ngrams import create_ngrams, create_ngram_counts


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    test_log = read_json_gz(args.test_log)['test']
    policies = list(test_log.keys())

    for policy_name in policies:

        if policy_name == 'most_freq':
            continue

        accuracy: List[float] = []
        mut_info: List[float] = []
        #ngram_mut_info: List[float] = []

        print('Num Rates: {}'.format(len(test_log[policy_name])))

        for rate, results in reversed(sorted(test_log[policy_name].items())):
            if args.dataset_order not in results:
                continue

            preds = np.array(results[args.dataset_order]['preds'])
            output_levels = np.array(results[args.dataset_order]['output_levels'])
            labels = np.array(results[args.dataset_order]['labels'])

            acc = compute_accuracy(predictions=preds, labels=labels)
            mi = compute_mutual_info(X=output_levels, Y=preds, should_normalize=False)

            #ngram_levels, ngram_preds = create_ngram_counts(levels=output_levels, preds=preds, n=n)
            #ngram_mi = compute_mutual_info(X=ngram_levels, Y=ngram_preds, should_normalize=False)

            accuracy.append(acc)
            mut_info.append(mi)
            #ngram_mut_info.append(ngram_mi)

        print('{} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\'.format(policy_name, np.average(accuracy), np.max(accuracy), np.average(mut_info), np.max(mut_info)))
