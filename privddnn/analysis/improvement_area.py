import numpy as np
import scipy.integrate
from argparse import ArgumentParser
from typing import List, Dict

from privddnn.analysis.read_logs import get_test_results
from privddnn.utils.metrics import compute_mutual_info


def area_between_curves(rates: List[str], upper: Dict[str, float], lower: Dict[str, float]):
    rate_values = list(sorted(map(float, rates)))
    upper_values: List[float] = []
    lower_values: List[float] = []

    for rate in rate_values:
        rate_str = str(round(rate, 2))
        upper_values.append(upper[rate_str])
        lower_values.append(lower[rate_str])

    upper_integral = scipy.integrate.trapz(y=upper_values, x=rate_values)
    lower_integral = scipy.integrate.trapz(y=lower_values, x=rate_values)
    return upper_integral - lower_integral


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log-folder', type=str, required=True)
    parser.add_argument('--metric', type=str, choices=['accuracy', 'mutual_information'], required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--trials', type=int, required=True)
    args = parser.parse_args()

    metric_results = get_test_results(folder_path=args.test_log_folder,
                                      fold='test',
                                      metric=args.metric,
                                      dataset_order=args.dataset_order,
                                      trials=args.trials)

    metric_scores: Dict[str, Dict[str, float]] = dict()
    for policy_name, policy_results in metric_results.items():
        metric_scores[policy_name] = dict()

        for rate, rate_results in policy_results.items():
            metric_scores[policy_name][rate] = np.average(rate_results)

    data_dependent_policies = ['max_prob']
    rates = list(metric_scores['random'].keys())

    for data_dependent_policy in data_dependent_policies:
        adaptive_random_policy = 'adaptive_random_{}'.format(data_dependent_policy)

        data_dependent_area = area_between_curves(rates=rates,
                                                  upper=metric_scores[data_dependent_policy],
                                                  lower=metric_scores['random'])
        adaptive_random_area = area_between_curves(rates=rates,
                                                   upper=metric_scores[adaptive_random_policy],
                                                   lower=metric_scores['random'])

        improvement_fraction = adaptive_random_area / data_dependent_area

        if args.metric == 'mutual_information':
            improvement_fraction = 1.0 - improvement_fraction

        print('{} & {:.4f}'.format(adaptive_random_policy, improvement_fraction))
