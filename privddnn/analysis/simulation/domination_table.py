import os
import numpy as np
from argparse import ArgumentParser
from typing import List, Dict, Tuple

from privddnn.analysis.utils.read_logs import get_summary_results
from privddnn.utils.inference_metrics import InferenceMetric
from privddnn.utils.constants import SMALL_NUMBER


def compare_policies(target_policies: List[str], baseline_policy: str, test_log_folder: str, dataset_order: str, should_print_counts: bool) -> List[Tuple[int, int, float]]:
    comparison: List[Tuple[int, int, float]] = []

    path_tokens = test_log_folder.split(os.sep)
    dataset_name = path_tokens[-3] if len(path_tokens[-1]) > 0 else path_tokens[-4]

    test_results = get_summary_results(folder_path=test_log_folder,
                                       dataset_order=dataset_order,
                                       fold='test',
                                       trials=1)
    accuracy_results = test_results[InferenceMetric.ACCURACY]

    for policy_name in target_policies:
        
        num_greater = 0
        total_rates = 0

        for rate in accuracy_results[policy_name].keys():
            # Do not compare at the edge of the range (all policies act the same)
            rate_values = list(map(float, rate.split(' ')))
            if abs(rate_values[0]) < SMALL_NUMBER or abs(rate_values[1]) < SMALL_NUMBER:
                continue

            target_accuracy_list = accuracy_results[policy_name][rate]
            target_accuracy = np.average(target_accuracy_list)

            baseline_accuracy_list = accuracy_results[baseline_policy][rate]
            baseline_accuracy = np.average(baseline_accuracy_list)

            num_greater += int(target_accuracy > baseline_accuracy)
            total_rates += 1

        comparison.append((num_greater, total_rates, (num_greater / total_rates) * 100.0))

    if should_print_counts:
        comparison_str = ' & '.join(map(lambda t: '{} / {} ({:.2f})'.format(t[0], t[1], t[2]), comparison))
    else:
        comparison_str = ' & '.join(map(lambda t: '{:.2f}'.format(t[2]), comparison))

    print('{} & {} \\\\'.format(dataset_name, comparison_str))

    return comparison


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log-folders', type=str, required=True, nargs='+')
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--target-policies', type=str, nargs='+', required=True)
    parser.add_argument('--baseline-policy', type=str, required=True)
    parser.add_argument('--should-print-counts', action='store_true')
    args = parser.parse_args()

    print('Dataset & {} \\\\'.format(' & '.join(args.target_policies)))

    aggregate = [(0, 0) for _ in args.target_policies]

    for test_log_folder in args.test_log_folders:
        comparison = compare_policies(target_policies=args.target_policies,
                                      baseline_policy=args.baseline_policy,
                                      test_log_folder=test_log_folder,
                                      dataset_order=args.dataset_order,
                                      should_print_counts=args.should_print_counts)

        for idx in range(len(aggregate)):
            curr = aggregate[idx]
            aggregate[idx] = (curr[0] + comparison[idx][0], curr[1] + comparison[idx][1])
    
    if args.should_print_counts:
        comparison_str = ' & '.join(map(lambda t: '{} / {} ({:.2f})'.format(t[0], t[1], t[0] / t[1]), aggregate))
    else:
        comparison_str = ' & '.join(map(lambda t: '{:.2f}'.format(t[0] / t[1]), comparison))

    print('Aggregate & {}'.format(comparison_str))

