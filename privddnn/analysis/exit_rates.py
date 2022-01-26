import numpy as np
from argparse import ArgumentParser
from typing import List
from privddnn.utils.file_utils import read_json_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    args = parser.parse_args()

    test_log = read_json_gz(args.test_log)['test']

    field_values: List[float] = []
    for rate, results in sorted(test_log[args.policy].items()):
        num_levels = results[args.dataset_order]['output_levels']

        print('Observed Exit Rate: {:6f} (Expected {})'.format(1.0 - np.average(num_levels), rate))
