import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List

from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.plotting import PLOT_STYLE, AXIS_FONT, TITLE_FONT, LABEL_FONT, LEGEND_FONT, POLICY_LABELS, DATASET_LABELS

WIDTH = 0.4


def compute_exit_rates(preds: List[int], output_levels: List[int]) -> List[float]:
    """
    Computes the exit rates stratified by each prediction.
    """
    num_labels = max(preds) + 1
    full_counts: List[int] = [0 for _ in range(num_labels)]
    total_counts: List[int] = [0 for _ in range(num_labels)]

    for pred, level in zip(preds, output_levels):
        full_counts[pred] += level
        total_counts[pred] += 1

    return [1.0 - (full / total) for full, total in zip(full_counts, total_counts)]


def get_dataset_name(path: str) -> str:
    tokens = path.split(os.sep)
    return tokens[-4] if len(tokens[-1]) == 0 else tokens[-3]


if __name__ == '__main__':
    parser = ArgumentParser('Script to compare exit behavior in black-box settings.')
    parser.add_argument('--target-log-folder', type=str, required=True, help='Path to the logs for the target model.')
    parser.add_argument('--substitute-log-folder', type=str, required=True, help='Path to the logs for the substitute model.')
    parser.add_argument('--policy', type=str, required=True, help='Name of the policy to compare.')
    parser.add_argument('--dataset-order', type=str, required=True, help='Name of the dataset order.')
    args = parser.parse_args()

    rate = 0.5
    exit_rate = '{:.2f} {:.2f}'.format(rate, 1.0 - rate)

    # Read the exact log files
    target_log_file = os.path.join(args.target_log_folder, '{}-trial0.json.gz'.format(args.policy))
    target_dataset_name = get_dataset_name(args.target_log_folder)
    target_log = read_json_gz(target_log_file)['test'][exit_rate][args.dataset_order]

    sub_log_file = os.path.join(args.substitute_log_folder, '{}-trial0.json.gz'.format(args.policy))
    sub_dataset_name = get_dataset_name(args.substitute_log_folder)
    sub_log = read_json_gz(sub_log_file)['val'][exit_rate][args.dataset_order]

    target_exit_rates = compute_exit_rates(preds=target_log['preds'], output_levels=target_log['output_levels'])
    sub_exit_rates = compute_exit_rates(preds=sub_log['preds'], output_levels=sub_log['output_levels'])

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        xs = np.arange(len(target_exit_rates))
        ax.bar(xs - (WIDTH / 2), target_exit_rates, width=WIDTH, label='Target AdNN ({})'.format(DATASET_LABELS[target_dataset_name]), color='#2c7fb8')
        ax.bar(xs + (WIDTH / 2), sub_exit_rates, width=WIDTH, label='Substitute AdNN ({})'.format(DATASET_LABELS[sub_dataset_name]), color='#810f7c')

        ax.axhline(rate, linestyle='dashed', color='k')
        ax.text(x=xs[-3], y=rate + 0.02, s='Target Exit Rate', fontsize=LABEL_FONT)

        ax.set_xticks(xs)
        ax.set_xticklabels(xs, fontsize=LABEL_FONT)
        ax.tick_params(axis='y', which='major', labelsize=LABEL_FONT)

        ax.legend(fontsize=LEGEND_FONT)

        ax.set_xlabel('Prediction', fontsize=AXIS_FONT)
        ax.set_ylabel('Early Exit Rate', fontsize=AXIS_FONT)
        ax.set_title('Early Exit Behavior Under {}'.format(POLICY_LABELS[args.policy]), fontsize=TITLE_FONT)

        plt.show()
