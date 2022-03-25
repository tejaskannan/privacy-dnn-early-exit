import os
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from typing import List

from privddnn.attack.attack_classifiers import MOST_FREQ, MAJORITY, LOGISTIC_REGRESSION_COUNT, NGRAM
from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.plotting import to_label, COLORS, AXIS_FONT, TITLE_FONT, LEGEND_FONT
from privddnn.utils.plotting import LINE_WIDTH, MARKER, MARKER_SIZE, PLOT_STYLE
from privddnn.dataset.dataset import Dataset

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True, choices=['l1_error', 'l2_error'])
    parser.add_argument('--attack-model', type=str, required=True, choices=[MAJORITY, LOGISTIC_REGRESSION_COUNT, MOST_FREQ, NGRAM, 'best'])
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--attack-train-log', type=str)
    parser.add_argument('--train-policy', type=str, default='same')
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    # Read the attack accuracy
    test_log = read_json_gz(args.test_log)

    attack_train_log = args.test_log if args.attack_train_log is None else args.attack_train_log
    attack_train_log_name = os.path.basename(attack_train_log)

    attack_key = '{}_{}'.format(attack_train_log_name.replace('_test-log.json.gz', ''), args.train_policy)
    attack_accuracy = test_log[attack_key][args.dataset_order]['input_attack_test']

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        rates = [round(r / 20.0, 2) for r in range(21)]

        for policy_name, attack_results in attack_accuracy.items():

            if args.attack_model == 'best':
                all_results: List[List[float]] = []
                for attack_result in attack_results.values():
                    all_results.append([r[args.metric] for r in attack_result])

                metric_results = np.max(all_results, axis=0).astype(float).tolist()
            else:
                metric_results = [r[args.metric] for r in attack_results[args.attack_model]]

            average_metric = sum(metric_results) / len(metric_results)
            min_metric = min(metric_results)

            print('{} & {:.2f} & {:.2f}'.format(policy_name, average_metric, min_metric))

            # The given rates are the fraction stopping at the first output (which makes the axes confusing)
            xs = [1.0 - float(r) for r in rates]

            ax.plot(xs, metric_results,
                    label=to_label(policy_name),
                    marker=MARKER,
                    markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH,
                    color=COLORS[policy_name])

        ax.set_xlabel('Frac Stopping at 2nd Output', fontsize=AXIS_FONT)
        ax.set_ylabel('Reconstruction Error', fontsize=AXIS_FONT)
        ax.set_title('{} Against Exit Policies'.format(args.metric.capitalize()), fontsize=TITLE_FONT)

        ax.legend(fontsize=LEGEND_FONT)

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
