from argparse import ArgumentParser

from privddnn.analysis.read_logs import get_input_attack_results
from privddnn.attack.attack_classifiers import MAJORITY
from privddnn.utils.constants import BIG_NUMBER


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log-folder', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True, choices=['l1_error', 'l2_error', 'weighted_l1_error'])
    parser.add_argument('--attack-model', type=str, required=True, choices=[MAJORITY])
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    # Read the input attack accuracy
    input_attack_results = get_input_attack_results(folder_path=args.test_log_folder,
                                                    fold='test',
                                                    dataset_order=args.dataset_order,
                                                    metric=args.metric,
                                                    attack_model=args.attack_model)

    for policy_name, attack_results in input_attack_results.items():

        metric_sum = 0.0
        metric_min = BIG_NUMBER
        metric_count = 0.0

        for rate, rate_results in attack_results.items():
            metric_sum += sum(rate_results)
            metric_min = min(metric_min, min(rate_results))
            metric_count += len(rate_results)

        average_metric = metric_sum / metric_count
        print('{} & {:.2f} & {:.2f}'.format(policy_name, average_metric, metric_min))
