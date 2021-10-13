import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List

from privddnn.dataset.dataset import Dataset
from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info, compute_geometric_mean
from privddnn.utils.plotting import to_label, COLORS, MARKER, MARKER_SIZE, LINE_WIDTH
from privddnn.utils.plotting import AXIS_FONT, TITLE_FONT, LABEL_FONT, LEGEND_FONT


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    test_log = read_json_gz(args.test_log)['test']
    policies = list(test_log.keys())

    tokens = args.test_log.split(os.sep)
    dataset = Dataset(tokens[-3])
    labels = dataset.get_test_labels()

    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))

        for policy_name in policies:

            if policy_name == 'most_freq':
                continue

            rates: List[float] = []
            accuracy: List[float] = []
            accuracy_std: List[float] = []
            mut_info: List[float] = []
            mut_info_std: List[float] = []

            for rate, results in reversed(sorted(test_log[policy_name].items())):

                rate_accuracy: List[float] = []
                rate_mi: List[float] = []

                for trial_result in results:
                    preds = np.array(trial_result['preds'])
                    output_levels = np.array(trial_result['output_levels'])

                    acc = compute_accuracy(predictions=preds, labels=labels)
                    mi = compute_mutual_info(X=output_levels, Y=preds, should_normalize=False)

                    rate_accuracy.append(acc)
                    rate_mi.append(mi)

                accuracy.append(np.average(rate_accuracy))
                accuracy_std.append(np.std(rate_accuracy))

                mut_info.append(np.average(rate_mi))
                mut_info_std.append(np.std(rate_mi))

                rates.append(round(1.0 - float(rate), 2))

            print('{} & {:.5f} & {:.5f} & {:.5f} & {:.5f}'.format(policy_name, np.average(accuracy), np.max(accuracy), np.average(mut_info), np.max(mut_info)))

            ax1.errorbar(rates, accuracy, yerr=accuracy_std, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=to_label(policy_name), color=COLORS[policy_name], capsize=3)
            ax2.errorbar(rates, mut_info, yerr=mut_info_std, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=to_label(policy_name), color=COLORS[policy_name], capsize=3)

        ax1.set_xlabel('Frac stopping at 2nd Exit', fontsize=AXIS_FONT)
        ax1.set_ylabel('Accuracy', fontsize=AXIS_FONT)
        ax1.set_title('Model Accuracy', fontsize=TITLE_FONT)
        ax1.legend(fontsize=LEGEND_FONT)
        ax1.tick_params(axis='both', which='major', labelsize=LABEL_FONT)

        ax2.set_xlabel('Frac stopping at 2nd Exit', fontsize=AXIS_FONT)
        ax2.set_ylabel('Mutual Information', fontsize=AXIS_FONT)
        ax2.set_title('Mut Info: Label vs Exit', fontsize=TITLE_FONT)
        ax2.tick_params(axis='both', which='major', labelsize=LABEL_FONT)

        plt.tight_layout()

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
