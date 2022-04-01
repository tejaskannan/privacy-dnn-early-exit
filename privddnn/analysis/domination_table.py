import numpy as np
from argparse import ArgumentParser
from typing import List, Dict

from privddnn.analysis.read_logs import get_test_results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log-folder', type=str, required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    args = parser.parse_args()

    accuracy_results = get_test_results(folder_path=args.test_log_folder,
                                        dataset_order=args.dataset_order,
                                        fold='test',
                                        metric='accuracy')

    policy_names = list(sorted(accuracy_results.keys()))
    print('& {}'.format(' & '.join(policy_names)))

    for first_policy in policy_names:
        print('{} &'.format(first_policy), end=' ')
        comparison: List[float] = []

        for second_policy in policy_names:

            num_greater = 0
            total_rates = 0
            for rate in accuracy_results[first_policy].keys():
                if np.isclose(float(rate), 0.0) or np.isclose(float(rate), 1.0):
                    continue

                first_accuracy_list = accuracy_results[first_policy][rate]
                first_accuracy = np.average(first_accuracy_list) - np.std(first_accuracy_list)

                second_accuracy_list = accuracy_results[second_policy][rate]
                second_accuracy = np.average(second_accuracy_list)

                num_greater += int(first_accuracy > second_accuracy)
                total_rates += 1

            comparison.append((num_greater, total_rates, num_greater / total_rates))

        print(' & '.join(map(lambda t: '{} / {} ({:.4f})'.format(t[0], t[1], t[2]), comparison)))
