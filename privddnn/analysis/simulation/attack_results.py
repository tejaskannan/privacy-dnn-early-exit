import os
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from typing import List

from privddnn.analysis.utils.read_logs import get_attack_results
from privddnn.attack.attack_classifiers import MOST_FREQ, MAJORITY, LOGISTIC_REGRESSION_COUNT, NGRAM, LOGISTIC_REGRESSION_NGRAM
from privddnn.utils.constants import BIG_NUMBER
from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.plotting import to_label, COLORS, AXIS_FONT, TITLE_FONT, LEGEND_FONT
from privddnn.utils.plotting import LINE_WIDTH, MARKER, MARKER_SIZE, PLOT_STYLE
from privddnn.dataset.dataset import Dataset

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log-folder', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True, choices=['accuracy', 'top2', 'top5', 'top10', 'top(k-1)', 'topUntil90', 'correct_rank'])
    parser.add_argument('--attack-model', type=str, required=True, choices=[MAJORITY, LOGISTIC_REGRESSION_COUNT, LOGISTIC_REGRESSION_NGRAM, MOST_FREQ, NGRAM, 'best'])
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--train-log-folder', type=str)
    parser.add_argument('--train-policy', type=str, default='same')
    parser.add_argument('--fold', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--should-plot', action='store_true')
    parser.add_argument('--trials', type=int)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    train_log_folder = args.test_log_folder if args.train_log_folder is None else args.train_log_folder

    # Read the attack accuracy
    attack_results = get_attack_results(folder_path=args.test_log_folder,
                                        fold=args.fold,
                                        dataset_order=args.dataset_order,
                                        attack_train_log=train_log_folder,
                                        attack_policy=args.train_policy,
                                        metric=args.metric,
                                        attack_model=args.attack_model,
                                        trials=args.trials)

    # Print out the aggregate results in a table format
    policy_names = list(sorted(attack_results.keys()))
    print('Policy & Avg {0} & Max {0} & Min {0} \\\\'.format(args.metric))

    for policy_name in policy_names:
        max_metric = 0.0
        min_metric = BIG_NUMBER
        metric_sums: List[float] = []  # Keep the sum of all metric values per trial

        num_rates = 0
        for rate, rate_results in attack_results[policy_name].items():
            if len(metric_sums) == 0:
                metric_sums = [0.0 for _ in rate_results]

            for idx in range(len(rate_results)):
                metric_sums[idx] += rate_results[idx]
                max_metric = max(max_metric, rate_results[idx])
                min_metric = min(min_metric, rate_results[idx])

            num_rates += 1

        metrics = [metric / num_rates for metric in metric_sums]

        avg_metric = np.average(metrics)
        std_metric = np.std(metrics)

        print('{} & {:.2f} ({:.2f}) & {:.2f} & {:.2f}  \\\\'.format(policy_name, avg_metric, std_metric, max_metric, min_metric))

    if args.should_plot:
        with plt.style.context(PLOT_STYLE):
            fig, ax = plt.subplots()

            for policy_name, policy_results in attack_results.items():
                rates = list(sorted(policy_results.keys()))
                metric_results = [policy_results[r] * 100.0 for r in rates]

                # The given rates are the fraction stopping at the first output (which makes the axes confusing)
                xs = [1.0 - float(r) for r in rates]

                ax.plot(xs, metric_results,
                        label=to_label(policy_name),
                        marker=MARKER,
                        markersize=MARKER_SIZE,
                        linewidth=LINE_WIDTH,
                        color=COLORS[policy_name])

            ax.set_xlabel('Frac Stopping at 2nd Output', fontsize=AXIS_FONT)
            ax.set_ylabel('Attack Accuracy (%)', fontsize=AXIS_FONT)
            ax.set_title('Attack {} Against Exit Policies'.format(args.metric.capitalize()), fontsize=TITLE_FONT)

            ax.legend(fontsize=LEGEND_FONT)

            if args.output_file is None:
                plt.show()
            else:
                plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
