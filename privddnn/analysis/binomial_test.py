import os
import numpy as np
import scipy.stats as stats
from argparse import ArgumentParser
from typing import List

from privddnn.dataset.dataset import Dataset
from privddnn.utils.file_utils import read_json_gz


def exec_test(output_levels: np.ndarray, labels: np.ndarray, target: float) -> List[float]:
    num_labels = np.max(labels) + 1
    level_counts = np.zeros(shape=(num_labels, ))
    total_counts = np.zeros_like(level_counts)

    for level, label in zip(output_levels, labels):
        level_counts[label] += level
        total_counts[label] += 1

    p_values: List[float] = []

    print(level_counts / total_counts)

    for label in range(num_labels):
        p_val = stats.binom_test(x=level_counts[label], n=total_counts[label], p=target, alternative='two-sided')
        p_values.append(p_val)

    return p_values


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    args = parser.parse_args()

    test_log = read_json_gz(args.test_log)['test']
    policies = list(test_log.keys())

    tokens = args.test_log.split(os.sep)
    dataset = Dataset(tokens[-3])
    labels = dataset.get_test_labels()

    for policy_name in policies:
        worst_p_values: List[float] = []

        for rate, results in reversed(sorted(test_log[policy_name].items())):
            output_levels = np.array(results['output_levels']).reshape(-1)
            target = 1.0 - float(rate)

            if target < 1e-7:
                continue

            test_results = exec_test(output_levels, labels=labels, target=target)
            worst_p_values.append(min(test_results))

        print('{} & {}'.format(policy_name, worst_p_values))
