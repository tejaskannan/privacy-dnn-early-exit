import os
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from typing import DefaultDict, List

from privddnn.exiting import ALL_POLICY_NAMES
from privddnn.utils.inference_metrics import InferenceMetric
from privddnn.analysis.utils.read_logs import get_summary_results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log-folders', type=str, required=True, nargs='+')
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--trials', type=int)
    args = parser.parse_args()

    policy_names = ALL_POLICY_NAMES

    aggregate_accuracy: DefaultDict[str, List[float]] = defaultdict(list)  # Policy -> [ aggregate accuracy per dataset ]
    aggregate_mut_info: DefaultDict[str, List[float]] = defaultdict(list)

    for test_log_folder in args.test_log_folders:
        dataset_results = get_summary_results(folder_path=test_log_folder,
                                              fold='test',
                                              dataset_order=args.dataset_order,
                                              trials=args.trials)

        accuracy_results = dataset_results[InferenceMetric.ACCURACY]
        mut_info_results = dataset_results[InferenceMetric.MUTUAL_INFORMATION]
 
        for policy_name in policy_names:
            accuracy_values: List[float] = []
            mut_info_values: List[float] = []

            for rate in sorted(accuracy_results[policy_name].keys()):
                avg_accuracy = np.mean(accuracy_results[policy_name][rate])
                accuracy_values.append(avg_accuracy)

                avg_mut_info = np.mean(mut_info_results[policy_name][rate])
                mut_info_values.append(avg_mut_info)

            aggregate_accuracy[policy_name].append(np.mean(accuracy_values))
            aggregate_mut_info[policy_name].append(np.max(mut_info_values))

    # Compute the aggregate results vs random
    print(' & Avg Accuracy Diff & Avg MI Diff \\')
    for policy_name in policy_names:
        acc_diff = [target - baseline for target, baseline in zip(aggregate_accuracy[policy_name], aggregate_accuracy['random'])] 
        mi_diff = [100.0 * (target - baseline) for target, baseline in zip(aggregate_mut_info[policy_name], aggregate_mut_info['random'])]

        print('{} & {:.2f} ({:.2f}) & {:.2f} ({:.2f})'.format(policy_name, np.mean(acc_diff), np.std(acc_diff), np.mean(mi_diff), np.std(mi_diff)))
