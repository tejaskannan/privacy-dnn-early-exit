import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from argparse import ArgumentParser
from collections import namedtuple
from typing import Dict, List

from privddnn.attack.attack_classifiers import LOGISTIC_REGRESSION_COUNT, LOGISTIC_REGRESSION_NGRAM, DECISION_TREE_COUNT, DECISION_TREE_NGRAM
from privddnn.exiting import ALL_POLICY_NAMES
from privddnn.utils.plotting import PLOT_STYLE, TITLE_FONT, LABEL_FONT, LEGEND_FONT, COLORS, AXIS_FONT
from privddnn.utils.plotting import POLICY_LABELS, DATASET_LABELS, MARKER
from privddnn.analysis.utils.read_logs import get_attack_results


matplotlib.rc('pdf', fonttype=42)
plt.rcParams['pdf.fonttype'] = 42


AttackResult = namedtuple('AttackResult', ['average', 'maximum'])
ADJUSTMENT = 3


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log-folders', type=str, required=True, nargs='+')
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--attack-model', type=str, required=True, choices=[DECISION_TREE_COUNT, DECISION_TREE_NGRAM, LOGISTIC_REGRESSION_COUNT, LOGISTIC_REGRESSION_NGRAM])
    parser.add_argument('--policies', type=str, required=True, nargs='+')
    parser.add_argument('--trials', type=int)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    policy_names = ALL_POLICY_NAMES if (len(args.policies) == 0) or ((len(args.policies) == 1) and args.policies[0] == 'all') else args.policies

    print('Number of Datasets: {}'.format(len(args.test_log_folders)))

    policy_results: Dict[str, List[AttackResult]] = dict()
    normalized_results: Dict[str, List[float]] = dict()
    raw_results: Dict[str, Dict[str, List[float]]] = dict()
    dataset_names: List[str] = []
    baseline_normalized: List[float] = []

    for policy_name in policy_names:
        policy_results[policy_name] = []
        normalized_results[policy_name] = []
        raw_results[policy_name] = dict()

    for test_log_folder in args.test_log_folders:
        path_tokens = test_log_folder.split(os.sep)
        dataset_name = path_tokens[-3] if len(path_tokens[-1]) > 0 else path_tokens[-4]
        dataset_label = DATASET_LABELS[dataset_name]
        dataset_names.append(dataset_label)

        # Read the attack results for this dataset
        attack_results = get_attack_results(folder_path=test_log_folder,
                                            fold='test',
                                            dataset_order=args.dataset_order,
                                            attack_train_log=test_log_folder,
                                            metric='accuracy',
                                            attack_model=args.attack_model,
                                            attack_policy='same',
                                            target_pred=None,
                                            trials=args.trials)

        baseline_results = get_attack_results(folder_path=test_log_folder,
                                            fold='test',
                                            dataset_order=args.dataset_order,
                                            attack_train_log=test_log_folder,
                                            metric='accuracy',
                                            attack_model='MostFrequent',
                                            attack_policy='same',
                                            target_pred=None,
                                            trials=args.trials)

        for policy_name in policy_names:
            accuracy_values: List[float] = []

            for rate, rate_results in attack_results[policy_name].items():
                avg_rate_result = np.average(rate_results)
                avg_base_result = np.average(baseline_results[policy_name][rate])

                normalized_results[policy_name].append(avg_rate_result / avg_base_result)
                accuracy_values.extend(rate_results)

            attack_result = AttackResult(average=np.average(accuracy_values), maximum=np.max(accuracy_values))
            policy_results[policy_name].append(attack_result)
            raw_results[policy_name][dataset_label] = accuracy_values

    # Compare the attack accuracies for each dataset
    for policy_name in policy_names:
        tstat, pval = stats.ttest_ind(normalized_results[policy_name], normalized_results['random'], equal_var=False)
        print('{} & {:.7f} & {:.7f}'.format(policy_name, np.mean(normalized_results[policy_name]), pval))

    # Holds the aggregate the results for all policies
    dataset_names.append('Avg')

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        num_policies = len(policy_names)
        width = 1.0 / (num_policies + 1)
        offset = -1 * width * (num_policies - 1) / 2
        xs = np.arange(len(dataset_names))

        print()

        for policy_name in policy_names:
            avg_accuracy = [res.average for res in policy_results[policy_name]]
            max_accuracy = [res.maximum for res in policy_results[policy_name]]
            base_accuracy = [res.maximum for res in policy_results['random']]

            aggregate = np.mean(max_accuracy)
            base_aggregate = np.mean(base_accuracy)
            aggregate_normalized = aggregate / base_aggregate

            print('{} & {:.4f} / {:.4f} & {:.4f}'.format(policy_name, aggregate, base_aggregate, aggregate_normalized))
            max_accuracy.append(aggregate)

            ax.bar(xs + offset, max_accuracy, width=width, label=POLICY_LABELS[policy_name], linewidth=1, edgecolor='black', color=COLORS[policy_name])

            offset += width

        ax.legend(fontsize=LEGEND_FONT, ncol=2, bbox_to_anchor=(0.49, 0.6), columnspacing=0.75)

        ax.set_xticks(xs)
        ax.set_xticklabels(dataset_names, fontsize=AXIS_FONT + ADJUSTMENT, rotation=45)

        dataset_order_name = 'Same Label' if args.dataset_order.startswith('same-label') else 'Nearest Neighbor'

        ax.tick_params(axis='y', labelsize=LEGEND_FONT + ADJUSTMENT)

        ax.set_xlabel('Dataset', fontsize=AXIS_FONT + ADJUSTMENT)
        ax.set_ylabel('Attack Accuracy (%)', fontsize=AXIS_FONT + ADJUSTMENT)
        ax.set_title('Maximum Attack Accuracy on {} Orders'.format(dataset_order_name), fontsize=TITLE_FONT + ADJUSTMENT)

        ax.axvline((xs[-1] + xs[-2]) / 2, linestyle='--', color='black', linewidth=2)

        plt.tight_layout()

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
