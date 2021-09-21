import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List

from privddnn.dataset import Dataset
from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.metrics import compute_avg_level_per_class, compute_max_likelihood_ratio



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--window-size', type=int, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    assert args.window_size > 0, 'Must provide a positive window size'

    test_log = read_json_gz(args.test_log)
    test_results = test_log['test']

    path_tokens = args.test_log.split(os.sep)
    dataset_name = path_tokens[-3]

    dataset = Dataset(dataset_name=dataset_name)

    # Get the test labels
    test_labels = dataset.get_test_labels()

    for strategy_name, strategy_results in test_results.items():

        factors: List[float] = []

        for rate_str, rate_results in strategy_results.items():
            rate = float(rate_str)

            # Get the avg level for each class, [K]
            avg_levels = compute_avg_level_per_class(output_levels=rate_results['output_levels'],
                                                     labels=test_labels)

            epsilon = compute_max_likelihood_ratio(label_stop_probs=avg_levels,
                                                   window_size=args.window_size)
            factors.append(epsilon)

        print(factors)
