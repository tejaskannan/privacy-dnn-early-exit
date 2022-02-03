import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List

from privddnn.dataset.dataset import Dataset
from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info, compute_geometric_mean
from privddnn.utils.ngrams import create_ngrams, create_ngram_counts
from privddnn.utils.plotting import to_label, COLORS, MARKER, MARKER_SIZE, LINE_WIDTH
from privddnn.utils.plotting import AXIS_FONT, TITLE_FONT, LABEL_FONT, LEGEND_FONT


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    test_log = read_json_gz(args.test_log)['test']
    policies = list(test_log.keys())
    n = 7

    #tokens = args.test_log.split(os.sep)
    #dataset = Dataset(tokens[-3])
    #labels = dataset.get_test_labels()

    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

        for policy_name in policies:

            if policy_name == 'most_freq':
                continue

            rates: List[float] = []
            accuracy: List[float] = []
            accuracy_std: List[float] = []
            mut_info: List[float] = []
            mut_info_std: List[float] = []
            ngram_mut_info: List[float] = []

            for rate, results in reversed(sorted(test_log[policy_name].items())):
                preds = np.array(results[args.dataset_order]['preds'])
                output_levels = np.array(results[args.dataset_order]['output_levels'])
                labels = np.array(results[args.dataset_order]['labels'])

                acc = compute_accuracy(predictions=preds, labels=labels)
                mi = compute_mutual_info(X=output_levels, Y=preds, should_normalize=False)

                ngram_levels, ngram_preds = create_ngram_counts(levels=output_levels, preds=preds, n=n)
                ngram_mi = compute_mutual_info(X=ngram_levels, Y=ngram_preds, should_normalize=False)

                accuracy.append(acc)
                mut_info.append(mi)
                ngram_mut_info.append(ngram_mi)

                rates.append(round(1.0 - float(rate), 2))

            print('{} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\'.format(policy_name, np.average(accuracy), np.max(accuracy), np.average(mut_info), np.max(mut_info), np.average(ngram_mut_info), np.max(ngram_mut_info)))

            ax1.plot(rates, accuracy, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=to_label(policy_name), color=COLORS[policy_name])
            ax2.plot(rates, mut_info, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=to_label(policy_name), color=COLORS[policy_name])
            ax3.plot(rates, ngram_mut_info, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=to_label(policy_name), color=COLORS[policy_name])

        ax1.set_xlabel('Frac stopping at 2nd Exit', fontsize=AXIS_FONT)
        ax1.set_ylabel('Accuracy', fontsize=AXIS_FONT)
        ax1.set_title('Model Accuracy', fontsize=TITLE_FONT)
        ax1.legend(fontsize=LEGEND_FONT)
        ax1.tick_params(axis='both', which='major', labelsize=LABEL_FONT)

        ax2.set_xlabel('Frac stopping at 2nd Exit', fontsize=AXIS_FONT)
        ax2.set_ylabel('Empirical Mutual Information (bits)', fontsize=AXIS_FONT)
        ax2.set_title('Mut Info: Label vs Exit', fontsize=TITLE_FONT)
        ax2.tick_params(axis='both', which='major', labelsize=LABEL_FONT)

        ax3.set_xlabel('Frac stopping at 2nd Exit', fontsize=AXIS_FONT)
        ax3.set_ylabel('Empirical Mutual Information (bits)', fontsize=AXIS_FONT)
        ax3.set_title('{}-gram Mut Info: Label vs Exit'.format(n), fontsize=TITLE_FONT)
        ax3.tick_params(axis='both', which='major', labelsize=LABEL_FONT)

        plt.tight_layout()

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
