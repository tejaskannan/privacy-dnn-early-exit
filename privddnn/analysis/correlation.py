"""
This script computes the correlation between the per-label stopping
rate on the validation and testing sets. A high correlation indicates
exploitable trends.
"""
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import Counter
from typing import List

from privddnn.dataset import Dataset
from privddnn.utils.file_utils import read_json_gz


def compute_avg_outputs(levels: List[int], labels: List[int]) -> List[float]:
    level_counter: Counter = Counter()
    label_counter: Counter = Counter()

    for level, label in zip(levels, labels):
        level_counter[label] += level
        label_counter[label] += 1

    results: List[float] = []
    for label in sorted(level_counter.keys()):
        results.append(level_counter[label] / label_counter[label])

    return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--rate', type=float, required=True)
    args = parser.parse_args()

    # Read the test log
    test_log = read_json_gz(args.test_log)

    # Read the validation and testing labels
    dataset_name = args.test_log.split(os.sep)[-3]
    dataset = Dataset(dataset_name)
    val_labels = dataset.get_val_labels()
    test_labels = dataset.get_test_labels()

    rate = str(round(args.rate, 2))

    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()

        for policy in sorted(test_log['val'].keys()):
            # Get the validation results for this rate
            val_outputs = test_log['val'][policy][rate]['output_levels']
            test_outputs = test_log['test'][policy][rate]['output_levels']

            # Compute the avg output level during both validation and testing
            val_avg_levels = compute_avg_outputs(levels=val_outputs, labels=val_labels)
            test_avg_levels = compute_avg_outputs(levels=test_outputs, labels=test_labels)

            # Compute the correlation between the two
            r, pvalue = stats.pearsonr(val_avg_levels, test_avg_levels)

            ax.scatter(val_avg_levels, test_avg_levels, label=policy)

            print('{} & {:.5f} & {:.5f}'.format(policy, r, pvalue))

        ax.set_xlabel('Validation Stopping Rate per Label')
        ax.set_ylabel('Test Stopping Rate per Label')
        ax.set_title('Relationship between Validation and Test Patterns')
        ax.legend()

        plt.show()
