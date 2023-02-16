import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Dict, DefaultDict

from privddnn.analysis.utils.read_logs import get_summary_results
from privddnn.utils.constants import SMALL_NUMBER
from privddnn.utils.inference_metrics import InferenceMetric


def area_between_curves(rates: List[str], upper: Dict[str, List[float]], lower: Dict[str, List[float]]):
    rate_values = list(sorted(map(lambda r: float(r.split(' ')[0]), rates)))
    upper_values: List[float] = []
    lower_values: List[float] = []

    for rate in rate_values:
        rate_str = '{:.2f} {:.2f}'.format(rate, 1.0 - rate)
        upper_values.append(np.median(upper[rate_str]))
        lower_values.append(np.median(lower[rate_str]))

    upper_integral = scipy.integrate.trapz(y=upper_values, x=rate_values)
    lower_integral = scipy.integrate.trapz(y=lower_values, x=rate_values)
    return upper_integral - lower_integral


def normalized_results(rates: List[str], target: Dict[str, List[float]], lower: Dict[str, List[float]], should_use_max: bool):
    results: List[float] = []

    if should_use_max:
        target_metric = max([np.average(target[r]) for r in rates])
        lower_metric = max([np.average(lower[r]) for r in rates])

        return target_metric - lower_metric
    else:
        for rate in rates:
            target_metric = np.average(target[rate])
            lower_metric = np.average(lower[rate])
            gain = (target_metric - lower_metric)

            results.append(gain)
        
        return results


def compute_improvement_scores(target_policy: str, baseline_policy: str, test_results: Dict[InferenceMetric, Dict[str, Dict[str, List[float]]]]) -> Dict[InferenceMetric, List[float]]:
    data_dependent_policy = 'max_prob' if target_policy.endswith('max_prob') else 'entropy'

    rates = list(test_results[InferenceMetric.ACCURACY]['random'].keys())

    #acc_data_dependent_area = area_between_curves(rates=rates,
    #                                              upper=test_results[InferenceMetric.ACCURACY][data_dependent_policy],
    #                                              lower=test_results[InferenceMetric.ACCURACY]['random'])
    #acc_target_area = area_between_curves(rates=rates,
    #                                      upper=test_results[InferenceMetric.ACCURACY][target_policy],
    #                                      lower=test_results[InferenceMetric.ACCURACY]['random'])

    #mi_data_dependent_area = area_between_curves(rates=rates,
    #                                             upper=test_results[InferenceMetric.MUTUAL_INFORMATION][data_dependent_policy],
    #                                             lower=test_results[InferenceMetric.MUTUAL_INFORMATION]['random'])
    #mi_target_area = area_between_curves(rates=rates,
    #                                     upper=test_results[InferenceMetric.MUTUAL_INFORMATION][target_policy],
    #                                     lower=test_results[InferenceMetric.MUTUAL_INFORMATION]['random'])

    #ngram_data_dependent_area = area_between_curves(rates=rates,
    #                                                upper=test_results[InferenceMetric.NGRAM_MI][data_dependent_policy],
    #                                                lower=test_results[InferenceMetric.NGRAM_MI]['random'])
    #ngram_target_area = area_between_curves(rates=rates,
    #                                        upper=test_results[InferenceMetric.NGRAM_MI][target_policy],
    #                                        lower=test_results[InferenceMetric.NGRAM_MI]['random'])


    #accuracy_score = acc_target_area / acc_data_dependent_area
    #mi_score = mi_target_area / mi_data_dependent_area
    #ngram_score = ngram_target_area / ngram_data_dependent_area

    accuracy_scores = normalized_results(rates=rates,
                                         target=test_results[InferenceMetric.ACCURACY][target_policy],
                                         lower=test_results[InferenceMetric.ACCURACY]['random'],
                                         should_use_max=False)
    mi_scores = normalized_results(rates=rates,
                                   target=test_results[InferenceMetric.MUTUAL_INFORMATION][target_policy],
                                   lower=test_results[InferenceMetric.MUTUAL_INFORMATION]['random'],
                                   should_use_max=True)
    ngram_scores = normalized_results(rates=rates,
                                      target=test_results[InferenceMetric.NGRAM_MI][target_policy],
                                      lower=test_results[InferenceMetric.NGRAM_MI]['random'],
                                      should_use_max=True)

    return {
        InferenceMetric.ACCURACY: accuracy_scores,
        InferenceMetric.MUTUAL_INFORMATION: [mi_scores],
        InferenceMetric.NGRAM_MI: [ngram_scores]
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log-folders', type=str, required=True, nargs='+')
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--target-policies', type=str, required=True, nargs='+')
    parser.add_argument('--baseline-policy', type=str, required=True)
    parser.add_argument('--trials', type=int)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    print('Num Datasets: {}'.format(len(args.test_log_folders)))
    print('==========')

    accuracy_scores: DefaultDict[str, List[float]] = defaultdict(list)
    mi_scores: DefaultDict[str, List[float]] = defaultdict(list)
    ngram_scores: DefaultDict[str, List[float]] = defaultdict(list)

    dataset_names: List[str] = []

    for test_log_folder in args.test_log_folders:
        test_results = get_summary_results(folder_path=test_log_folder,
                                           fold='test',
                                           dataset_order=args.dataset_order,
                                           trials=args.trials)

        for target_policy in args.target_policies:
            scores = compute_improvement_scores(target_policy=target_policy,
                                                baseline_policy=args.baseline_policy,
                                                test_results=test_results)

            accuracy_scores[target_policy].extend(scores[InferenceMetric.ACCURACY])
            mi_scores[target_policy].extend(scores[InferenceMetric.MUTUAL_INFORMATION])
            ngram_scores[target_policy].extend(scores[InferenceMetric.NGRAM_MI])

    for target_policy in args.target_policies:
        avg_accuracy = np.average(accuracy_scores[target_policy])
        std_accuracy = np.std(accuracy_scores[target_policy])

        max_mi = np.max(mi_scores[target_policy])
        #std_mi = np.std(mi_scores[target_policy])

        max_ngram = np.max(ngram_scores[target_policy])
        #std_ngram = np.std(ngram_scores[target_policy])

        #print('Policy: {}, Acc Score: {:.4f} ({:.4f}), MI Score: {:.4f} ({:.4f}), Ngram Score: {:.4f} ({:.4f})'.format(target_policy, avg_accuracy, std_accuracy, avg_mi, std_mi, avg_ngram, std_ngram))
        print('Policy: {}, Acc Score: {:.4f} ({:.4f}), MI Score: {:.4f}, Ngram Score: {:.4f}'.format(target_policy, avg_accuracy, std_accuracy, max_mi, max_ngram))
