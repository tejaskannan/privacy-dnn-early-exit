import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Dict, DefaultDict

from privddnn.analysis.utils.read_logs import get_summary_results
from privddnn.utils.inference_metrics import InferenceMetric
from privddnn.utils.plotting import PLOT_STYLE, to_label, TITLE_FONT, AXIS_FONT, LABEL_FONT, LEGEND_FONT, DATASET_LABELS, POLICY_LABELS


MI_COLORS = {
    'max_prob': '#ca0020',
    'entropy': '#f4a582'
}

ACCURACY_COLORS = {
    'max_prob': '#0571b0',
    'entropy': '#92c5de'
}


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


def compute_improvement_scores(target_policy: str, baseline_policy: str, test_results: Dict[InferenceMetric, Dict[str, Dict[str, List[float]]]]) -> Dict[InferenceMetric, float]:
    data_dependent_policy = 'max_prob' if target_policy.endswith('max_prob') else 'entropy'

    rates = list(test_results[InferenceMetric.ACCURACY]['random'].keys())

    acc_data_dependent_area = area_between_curves(rates=rates,
                                                  upper=test_results[InferenceMetric.ACCURACY][data_dependent_policy],
                                                  lower=test_results[InferenceMetric.ACCURACY]['random'])
    acc_target_area = area_between_curves(rates=rates,
                                          upper=test_results[InferenceMetric.ACCURACY][target_policy],
                                          lower=test_results[InferenceMetric.ACCURACY]['random'])

    mi_data_dependent_area = area_between_curves(rates=rates,
                                                 upper=test_results[InferenceMetric.MUTUAL_INFORMATION][data_dependent_policy],
                                                 lower=test_results[InferenceMetric.MUTUAL_INFORMATION]['random'])
    mi_target_area = area_between_curves(rates=rates,
                                         upper=test_results[InferenceMetric.MUTUAL_INFORMATION][target_policy],
                                         lower=test_results[InferenceMetric.MUTUAL_INFORMATION]['random'])

    accuracy_score = acc_target_area / acc_data_dependent_area
    mi_score = 1.0 - (mi_target_area / mi_data_dependent_area)

    return {
        InferenceMetric.ACCURACY: accuracy_score,
        InferenceMetric.MUTUAL_INFORMATION: mi_score
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

    accuracy_scores: DefaultDict[str, List[float]] = defaultdict(list)
    mi_scores: DefaultDict[str, List[float]] = defaultdict(list)

    dataset_names: List[str] = []

    for test_log_folder in args.test_log_folders:
        path_tokens = test_log_folder.split(os.sep)
        dataset_name = path_tokens[-3] if len(path_tokens[-1]) > 0 else path_tokens[-4]
        dataset_names.append(DATASET_LABELS[dataset_name])

        test_results = get_summary_results(folder_path=test_log_folder,
                                           fold='test',
                                           dataset_order=args.dataset_order,
                                           trials=args.trials)

        for target_policy in args.target_policies:
            scores = compute_improvement_scores(target_policy=target_policy,
                                                baseline_policy=args.baseline_policy,
                                                test_results=test_results)

            accuracy_scores[target_policy].append(scores[InferenceMetric.ACCURACY])
            mi_scores[target_policy].append(scores[InferenceMetric.MUTUAL_INFORMATION])

    for target_policy in args.target_policies:
        aggregate_accuracy = np.median(accuracy_scores[target_policy])
        aggregate_mi = np.median(mi_scores[target_policy])

        accuracy_scores[target_policy].append(aggregate_accuracy)
        mi_scores[target_policy].append(aggregate_mi)
    
    dataset_names.append('All')

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        xs = np.arange(len(dataset_names))
        width = 0.15
        num_policies = len(args.target_policies)
        offset = -(num_policies - 0.5) * width

        for target_policy in args.target_policies:
            color_policy = 'max_prob' if target_policy.endswith('max_prob') else 'entropy'

            ax.bar(xs + offset, accuracy_scores[target_policy], width=width, label='{} Accuracy'.format(POLICY_LABELS[target_policy]), color=ACCURACY_COLORS[color_policy])
            ax.bar(xs + offset + num_policies * width, mi_scores[target_policy], width=width, label='{} Mut Info'.format(POLICY_LABELS[target_policy]), color=MI_COLORS[color_policy])

            #for x, accuracy_score in zip(xs + offset, accuracy_scores[target_policy]):
            #    x_offset = 0.1 if (x % 2) == 1 else 0.2

            #    ax.annotate('{:.3f}'.format(accuracy_score),
            #                xy=(x, accuracy_score),
            #                xytext=(x - x_offset, accuracy_score + 0.05),
            #                fontsize=10)

            offset += width

        ax.set_xticks(xs)
        ax.set_xticklabels(labels=dataset_names, fontsize=LABEL_FONT, rotation=40)

        ax.axvline((xs[-1] + xs[-2]) / 2.0, linestyle='--', color='k')

        ax.set_xlabel('Dataset', size=AXIS_FONT)
        ax.set_ylabel('Improvement Area', size=AXIS_FONT)

        ax.set_title('Relative Improvement Compared to {} Exiting'.format(args.baseline_policy.capitalize()), size=TITLE_FONT)
        ax.legend(bbox_to_anchor=(0.5, -0.05), fontsize=LEGEND_FONT, loc='upper center', fancybox=True, ncol=2, facecolor='white', framealpha=1)

        plt.tight_layout()

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, transparent=True, bbox_inches='tight')
