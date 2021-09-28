"""
This script computes the expected KL divergence of the per-label stopping
rate between the validation and testing sets. A low divergence indicates
exploitable trend from validation to testing.
"""
import scipy.stats as stats
import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import Counter
from typing import List

from privddnn.dataset import Dataset
from privddnn.utils.constants import SMALL_NUMBER
from privddnn.utils.file_utils import read_json_gz


def compute_avg_outputs(levels: List[int], labels: List[int]) -> np.ndarray:
    num_labels = np.max(labels) + 1
    level_counts = np.zeros(shape=(num_labels, 2))
    total_counts = np.zeros(shape=(num_labels, 1))

    for level, label in zip(levels, labels):
        level_counts[label, level] += 1
        total_counts[label, 0] += 1

    return level_counts / total_counts


def binary_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    assert p.shape[0] == 2 and q.shape[0] == 2, 'Must provide 2-class inputs'

    kl_div = 0.0
    for p_val, q_val in zip(p, q):
        kl_div += p_val * np.log(p_val / (q_val + SMALL_NUMBER) + SMALL_NUMBER)

    return kl_div


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    args = parser.parse_args()

    # Read the test log
    test_log = read_json_gz(args.test_log)

    # Read the validation and testing labels
    dataset_name = args.test_log.split(os.sep)[-3]
    dataset = Dataset(dataset_name)
    val_labels = dataset.get_val_labels()
    test_labels = dataset.get_test_labels()

    label_counts = np.bincount(test_labels)
    label_freq = label_counts / np.sum(label_counts)

    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()

        for policy in sorted(test_log['val'].keys()):

            divergences: List[float] = []
            data_ranges: List[float] = []

            for rate in sorted(test_log['val'][policy].keys()):
                if rate in ('0.0', '1.0'):
                    continue

                # Get the validation results for this rate
                val_outputs = test_log['val'][policy][rate]['output_levels']
                test_outputs = test_log['test'][policy][rate]['output_levels']

                # Compute the avg output level during both validation and testing
                val_avg_levels = compute_avg_outputs(levels=val_outputs, labels=val_labels)
                test_avg_levels = compute_avg_outputs(levels=test_outputs, labels=test_labels)

                data_range = np.max(test_avg_levels[:, 0]) - np.min(test_avg_levels[:, 0])

                correlation, pvalue = stats.pearsonr(val_avg_levels[: ,0], test_avg_levels[:, 0])

                #expected_div = 0.0
                #for label in range(len(label_counts)):
                #    #kl_div = binary_kl_divergence(val_avg_levels[label], test_avg_levels[label])
                #    div = np.linalg.norm(val_avg_levels[label] - test_avg_levels[label], ord=1)
                #    expected_div += label_freq[label] * div
                #expected_div = np.linalg.norm(val_avg_levels[:, 0] - test_avg_levels[:, 0], ord=1)
                #expected_div /= data_range

                divergences.append((correlation, pvalue))
                data_ranges.append(data_range)

            print('{} & {}'.format(policy, divergences))
            print('{} & {}'.format(policy, data_ranges))
            print('==========')
