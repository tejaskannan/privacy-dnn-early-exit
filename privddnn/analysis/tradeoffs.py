import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List

from privddnn.dataset.dataset import Dataset
from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info, compute_geometric_mean
from privddnn.utils.ngrams import create_ngrams, create_ngram_counts
from privddnn.utils.plotting import to_label, COLORS, MARKER, MARKER_SIZE, LINE_WIDTH, CAPSIZE
from privddnn.utils.plotting import AXIS_FONT, TITLE_FONT, LABEL_FONT, LEGEND_FONT
from privddnn.analysis.read_logs import get_test_results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log-folder', type=str, required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--trials', type=int)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    # Read the test results
    accuracy_results = get_test_results(folder_path=args.test_log_folder,
                                        fold='test',
                                        dataset_order=args.dataset_order,
                                        metric='accuracy',
                                        trials=args.trials)

    mut_info_results = get_test_results(folder_path=args.test_log_folder,
                                        fold='test',
                                        dataset_order=args.dataset_order,
                                        metric='mutual_information',
                                        trials=args.trials)

    exit_deviation_results = get_test_results(folder_path=args.test_log_folder,
                                              fold='test',
                                              dataset_order=args.dataset_order,
                                              metric='exit_rate_deviation',
                                              trials=args.trials)

    avg_exit_results = get_test_results(folder_path=args.test_log_folder,
                                        fold='test',
                                        dataset_order=args.dataset_order,
                                        metric='avg_exit',
                                        trials=args.trials)

    ngram_size = 3
    ngram_results = get_test_results(folder_path=args.test_log_folder,
                                     fold='test',
                                     dataset_order=args.dataset_order,
                                     metric='ngram_{}'.format(ngram_size),
                                     trials=args.trials)

    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

        for policy_name in sorted(accuracy_results.keys()):

            rates: List[float] = []
            accuracy_list: List[float] = []
            accuracy_std_list: List[float] = []
            mut_info_list: List[float] = []
            mut_info_std_list: List[float] = []
            ngram_mut_info_list: List[float] = []
            ngram_mut_info_std_list: List[float] = []

            for rate in sorted(accuracy_results[policy_name].keys()):
                # Add the accuracy results
                avg_accuracy = np.average(accuracy_results[policy_name][rate])
                std_accuracy = np.std(accuracy_results[policy_name][rate])

                accuracy_list.append(avg_accuracy)
                accuracy_std_list.append(std_accuracy)

                # Add the mutual information results
                avg_mut_info = np.average(mut_info_results[policy_name][rate])
                std_mut_info = np.std(mut_info_results[policy_name][rate])
                
                mut_info_list.append(avg_mut_info)
                mut_info_std_list.append(std_mut_info)

                # Add the ngram mutual information results
                avg_ngram = np.average(ngram_results[policy_name][rate])
                std_ngram = np.std(ngram_results[policy_name][rate])

                ngram_mut_info_list.append(avg_ngram)
                ngram_mut_info_std_list.append(std_ngram)

                exit_rate = np.average(avg_exit_results[policy_name][rate])

                rates.append(exit_rate)
                num_trials = len(accuracy_results[policy_name][rate])

            # Get the deviation for the average result across all trials
            avg_accuracy_list: List[float] = []
            avg_mut_info_list: List[float] = []
            avg_ngram_list: List[float] = []

            for trial in range(num_trials):
                trial_accuracy: List[float] = []
                trial_mut_info: List[float] = []
                trial_ngram: List[float] = []

                for rate in sorted(accuracy_results[policy_name].keys()):
                    trial_accuracy.append(accuracy_results[policy_name][rate][trial])
                    trial_mut_info.append(mut_info_results[policy_name][rate][trial])
                    trial_ngram.append(ngram_results[policy_name][rate][trial])

                avg_accuracy_list.append(np.average(trial_accuracy))
                avg_mut_info_list.append(np.average(trial_mut_info))
                avg_ngram_list.append(np.average(trial_ngram))

            # Print the aggregate results
            avg_acc = np.average(accuracy_list)
            max_acc = np.max(accuracy_list)
            std_acc = np.std(avg_accuracy_list)

            avg_mi = np.average(mut_info_list)
            max_mi = np.max(mut_info_list)
            std_mi = np.std(avg_mut_info_list)

            avg_ngram = np.average(ngram_mut_info_list)
            max_ngram = np.max(ngram_mut_info_list)
            std_ngram = np.std(avg_ngram_list)

            print('{} & {:.4f} ({:.4f}) & {:.4f} & {:.4f} ({:.4f}) & {:.4f} & {:.4f} ({:.4f}) & {:.4f} \\\\'.format(policy_name, avg_acc, std_acc, max_acc, avg_mi, std_mi, max_mi, avg_ngram, std_ngram, max_ngram))

            # Plot the results
            ax1.errorbar(rates, accuracy_list, yerr=accuracy_std_list, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=to_label(policy_name), color=COLORS[policy_name], capsize=CAPSIZE)
            ax2.errorbar(rates, mut_info_list, yerr=mut_info_std_list, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=to_label(policy_name), color=COLORS[policy_name], capsize=CAPSIZE)
            ax3.errorbar(rates, ngram_mut_info_list, yerr=ngram_mut_info_std_list, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=to_label(policy_name), color=COLORS[policy_name], capsize=CAPSIZE)

        ax1.set_xlabel('Average Exit Point', fontsize=AXIS_FONT)
        ax1.set_ylabel('Accuracy', fontsize=AXIS_FONT)
        ax1.set_title('Model Accuracy', fontsize=TITLE_FONT)
        ax1.legend(fontsize=LEGEND_FONT)
        ax1.tick_params(axis='both', which='major', labelsize=LABEL_FONT)

        ax2.set_xlabel('Average Exit Point', fontsize=AXIS_FONT)
        ax2.set_ylabel('Empirical Mutual Information (bits)', fontsize=AXIS_FONT)
        ax2.set_title('Mut Info: Label vs Exit', fontsize=TITLE_FONT)
        ax2.tick_params(axis='both', which='major', labelsize=LABEL_FONT)

        ax3.set_xlabel('Average Exit Point', fontsize=AXIS_FONT)
        ax3.set_ylabel('Empirical Mutual Information (bits)', fontsize=AXIS_FONT)
        ax3.set_title('{}-gram Mut Info: Label vs Exit'.format(ngram_size), fontsize=TITLE_FONT)
        ax3.tick_params(axis='both', which='major', labelsize=LABEL_FONT)

        plt.tight_layout()

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
