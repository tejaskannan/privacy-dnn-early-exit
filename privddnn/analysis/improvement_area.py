import numpy as np
import scipy.integrate
from argparse import ArgumentParser
from typing import List, Dict
from privddnn.utils.file_utils import read_json_gz


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
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    args = parser.parse_args()

    test_log = read_json_gz(args.test_log)['test']
    accuracy_scores: Dict[str, Dict[str, float]] = dict()

    for policy_name, policy_results in sorted(test_log.items()):
        policy_scores: Dict[str, float] = dict()

        for rate, rate_results in policy_results.items():
            preds = rate_results[args.dataset_order]['preds']
            labels = rate_results[args.dataset_order]['labels']
            accuracy = np.average(np.isclose(preds, labels).astype(float))
            policy_scores[rate] = accuracy

        accuracy_scores[policy_name] = policy_scores

    data_dependent_policies = ['max_prob']
    rates = list(accuracy_scores['random'].keys())

    for data_dependent_policy in data_dependent_policies:
        adaptive_random_policy = 'adaptive_random_{}'.format(data_dependent_policy)

        data_dependent_area = area_between_curves(rates=rates,
                                                  upper=accuracy_scores[data_dependent_policy],
                                                  lower=accuracy_scores['random'])
        adaptive_random_area = area_between_curves(rates=rates,
                                                   upper=accuracy_scores[adaptive_random_policy],
                                                   lower=accuracy_scores['random'])

        improvement_fraction = adaptive_random_area / data_dependent_area

        print('{} & {:.4f}'.format(adaptive_random_policy, improvement_fraction))
