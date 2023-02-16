import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from argparse import ArgumentParser
from collections import namedtuple
from typing import Dict, List

from privddnn.attack.attack_classifiers import LOGISTIC_REGRESSION_COUNT, LOGISTIC_REGRESSION_NGRAM
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
    parser.add_argument('--attack-model', type=str, required=True, choices=[LOGISTIC_REGRESSION_COUNT, LOGISTIC_REGRESSION_NGRAM])
    parser.add_argument('--policies', type=str, required=True, nargs='+')
    parser.add_argument('--trials', type=int)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    print('Number of Datasets: {}'.format(len(args.test_log_folders)))

    policy_results: Dict[str, List[AttackResult]] = dict()
    normalized_results: Dict[str, List[float]] = dict()
    raw_results: Dict[str, List[float]] = dict()
    dataset_names: List[str] = []
    baseline_normalized: List[float] = []

    for policy_name in args.policies:
        policy_results[policy_name] = []
        normalized_results[policy_name] = []
        raw_results[policy_name] = []

    for test_log_folder in args.test_log_folders:
        path_tokens = test_log_folder.split(os.sep)
        dataset_name = path_tokens[-3] if len(path_tokens[-1]) > 0 else path_tokens[-4]
        dataset_names.append(DATASET_LABELS[dataset_name])

        # Read the attack results for this dataset
        attack_results = get_attack_results(folder_path=test_log_folder,
                                            fold='test',
                                            dataset_order=args.dataset_order,
                                            attack_train_log=test_log_folder,
                                            metric='accuracy',
                                            attack_model=args.attack_model,
                                            attack_policy='same',
                                            trials=args.trials)

        for policy_name in args.policies:
            accuracy_values: List[float] = []

            for rate, rate_results in attack_results[policy_name].items():
                accuracy_values.extend(rate_results)

                avg_rate_result = np.average(rate_results)
                avg_base_result = np.average(attack_results['random'][rate])
                normalized_results[policy_name].append(avg_rate_result / avg_base_result)

                if policy_name == 'random':
                    baseline_normalized.append(rate_results[0] / avg_base_result)

            attack_result = AttackResult(average=np.average(accuracy_values), maximum=np.max(accuracy_values))
            policy_results[policy_name].append(attack_result)
            raw_results[policy_name].extend(accuracy_values)

    # Aggregate the results for all policies
    dataset_names.append('All')

    for policy_name in args.policies:
        avg_accuracy = np.average(raw_results[policy_name])
        max_accuracy = np.max(raw_results[policy_name])

        normalized = normalized_results[policy_name]
        avg_normalized = np.average(normalized)
        std_normalized = np.std(normalized)

        print('{}. Avg Normalized -> {:.4f} ({:.4f})'.format(policy_name, avg_normalized, std_normalized))

        t_stat, pvalue = stats.ttest_ind(normalized, baseline_normalized, equal_var=False)

        #t_stat, pvalue = stats.ttest_ind(raw_results[policy_name], raw_results['random'], equal_var=False)
        print('\tt-test pvalue: {:.4f}, # samples: {}'.format(pvalue, len(normalized)))

        policy_results[policy_name].append(AttackResult(average=avg_accuracy, maximum=max_accuracy))

    #with plt.style.context(PLOT_STYLE):
    #    fig, ax = plt.subplots()

    #    ax.hist(x=baseline_normalized, bins=25)
    #    ax.set_title('Random (one trial) Normalized Attack Accuracy')

    #    plt.savefig('random_normalized.pdf')

    with plt.style.context(PLOT_STYLE):
        # Full uses (9, 4.5)
        # Nearest uses 

        fig, ax = plt.subplots(figsize=(8, 5))

        num_policies = len(args.policies)
        width = 1.0 / (num_policies + 1)
        offset = -1 * width * (num_policies - 1) / 2
        xs = np.arange(len(dataset_names))

        for policy_name in args.policies:
            avg_accuracy = [res.average for res in policy_results[policy_name]]
            max_accuracy = [res.maximum for res in policy_results[policy_name]]

            ax.bar(xs + offset, avg_accuracy, width=width, label=POLICY_LABELS[policy_name], linewidth=1, edgecolor='black', color=COLORS[policy_name])
            ax.scatter(xs + offset, max_accuracy, color=COLORS[policy_name], marker=MARKER, s=16)

            offset += width

        ax.legend(fontsize=LEGEND_FONT, ncol=2)

        ax.set_xticks(xs)
        ax.set_xticklabels(dataset_names, fontsize=AXIS_FONT + ADJUSTMENT, rotation=30)

        dataset_order_name = 'Same Label' if args.dataset_order.startswith('same-label') else 'Nearest Neighbor'

        #yticklabels = list(map(int, ax.get_yticklabels()))
        #ax.set_yticklabels(yticklabels, fontsize=LEGEND_FONT)
        ax.tick_params(axis='y', labelsize=LEGEND_FONT + ADJUSTMENT)

        ax.set_xlabel('Dataset', fontsize=AXIS_FONT + ADJUSTMENT)
        ax.set_ylabel('Attack Accuracy (%)', fontsize=AXIS_FONT + ADJUSTMENT)
        #ax.set_title('Average and Maximum Attack Accuracy for {}'.format(dataset_order_name), fontsize=TITLE_FONT + ADJUSTMENT)
        ax.set_title('Average and Maximum Attack Accuracy', fontsize=TITLE_FONT + ADJUSTMENT)

        ax.axvline((xs[-1] + xs[-2]) / 2, linestyle='--', color='black', linewidth=2)

        plt.tight_layout()

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
