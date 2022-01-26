import numpy as np
from argparse import ArgumentParser
from typing import List
from privddnn.utils.file_utils import read_json_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--field', type=str, required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    args = parser.parse_args()

    test_log = read_json_gz(args.test_log)['test']

    for policy_name in test_log.keys():

        field_values: List[float] = []
        exit_rates: List[str] = []

        for rate, results in sorted(test_log[policy_name].items()):
            num_samples = len(results[args.dataset_order]['preds'])

            if args.field == 'num_changed':
                value = results[args.dataset_order][args.field]
            elif args.field == 'exit_rate':
                value = np.sum([(level ^ 1) for level in results[args.dataset_order]['output_levels']])
            else:
                value = results[args.dataset_order]['selection_counts'][args.field.upper()]

            field_values.append(value / num_samples)
            exit_rates.append(rate)

        print(' & {}'.format(' & '.join(exit_rates)))
        print('{} & {}'.format(policy_name, ' & '.join(map(lambda s: '{:.5f}'.format(s), field_values))))
