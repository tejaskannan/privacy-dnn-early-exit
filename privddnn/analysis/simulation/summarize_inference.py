import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, DefaultDict

from privddnn.utils.plotting import COLORS, MARKER, MARKER_SIZE, LINE_WIDTH, CAPSIZE, PLOT_STYLE
from privddnn.utils.plotting import AXIS_FONT, TITLE_FONT, LABEL_FONT, LEGEND_FONT, POLICY_LABELS
from privddnn.utils.inference_metrics import InferenceMetric
from privddnn.analysis.utils.read_logs import get_summary_results


POLICIES = ['random', 'entropy', 'label_entropy', 'cgr_entropy', 'max_prob', 'label_max_prob', 'cgr_max_prob']
ADJUSTMENT = 3


matplotlib.rc('pdf', fonttype=42)
plt.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log-folder', type=str, required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--trials', type=int)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    # Read the test results
    window_size = int(args.dataset_order.split('-')[-1])
    test_results = get_summary_results(folder_path=args.test_log_folder,
                                       fold='test',
                                       dataset_order=args.dataset_order,
                                       trials=args.trials)

    accuracy_results = test_results[InferenceMetric.ACCURACY]
    mut_info_results = test_results[InferenceMetric.MUTUAL_INFORMATION]
    avg_exit_results = test_results[InferenceMetric.AVG_EXIT]

    accuracy_agg: List[float] = []
    accuracy_std_agg: List[float] = []

    mut_info_agg: List[float] = []
    mut_info_std_agg: List[float] = []
    mut_info_max_agg: List[float] = []

    with plt.style.context(PLOT_STYLE):
        #fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
        fig, ax = plt.subplots(figsize=(6, 4))

        policy_names = [policy_name for policy_name in POLICIES if policy_name in accuracy_results]

        print('Metric & {} \\\\'.format(' & '.join(policy_names)))
        #print('Policy & Avg (Std) Acc & Max Acc & Avg (Std) MI & Max MI & Avg (Std) {0}-Gram MI & Max {0}-Gram MI \\\\'.format(window_size))

        for policy_name in policy_names:

            rates: List[float] = []
            accuracy_list: List[float] = []
            accuracy_std_list: List[float] = []
            mut_info_list: List[float] = []
            mut_info_std_list: List[float] = []

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

                exit_rate = np.average(avg_exit_results[policy_name][rate])

                rates.append(exit_rate)
                num_trials = len(accuracy_results[policy_name][rate])

            # Get the deviation for the average result across all trials
            avg_accuracy_list: List[float] = []
            avg_mut_info_list: List[float] = []

            for trial in range(num_trials):
                trial_accuracy: List[float] = []
                trial_mut_info: List[float] = []

                for rate in sorted(accuracy_results[policy_name].keys()):
                    trial_accuracy.append(accuracy_results[policy_name][rate][trial])
                    trial_mut_info.append(mut_info_results[policy_name][rate][trial])

                avg_accuracy_list.append(np.average(trial_accuracy))
                avg_mut_info_list.append(np.average(trial_mut_info))

            # Collect the aggregate results
            avg_acc = np.average(accuracy_list)
            max_acc = np.max(accuracy_list)
            std_acc = np.std(avg_accuracy_list)

            avg_mi = np.average(mut_info_list)
            max_mi = np.max(mut_info_list)
            std_mi = np.std(avg_mut_info_list)

            accuracy_agg.append(avg_acc)
            accuracy_std_agg.append(std_acc)

            mut_info_agg.append(avg_mi)
            mut_info_std_agg.append(std_mi)
            mut_info_max_agg.append(max_mi)

            # Plot the results
            ax.errorbar(rates, accuracy_list, yerr=accuracy_std_list, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=POLICY_LABELS[policy_name], color=COLORS[policy_name], capsize=CAPSIZE)

        ax.set_xlabel('Fraction of Inputs using the Full Model', fontsize=AXIS_FONT + ADJUSTMENT)
        ax.set_ylabel('Inference Accuracy (%)', fontsize=AXIS_FONT + ADJUSTMENT)
        ax.set_title('Inference Accuracy for Target Exit Rates', fontsize=TITLE_FONT + ADJUSTMENT)
        ax.legend(fontsize=LEGEND_FONT)
        ax.tick_params(axis='both', which='major', labelsize=LABEL_FONT + ADJUSTMENT - 0.5)

        plt.tight_layout()

        # Print the result table
        print('Accuracy & {}'.format(' & '.join(map(lambda t: '{:.2f} ({:.2f})'.format(t[0], t[1]), zip(accuracy_agg, accuracy_std_agg)))))
        print('Mut Info & {}'.format(' & '.join(map(lambda t: '{:.2f}'.format(t * 100.0), mut_info_max_agg))))

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
