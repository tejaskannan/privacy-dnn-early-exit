import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List

from privddnn.utils.file_utils import read_json
from privddnn.utils.plotting import PLOT_STYLE, AXIS_FONT, TITLE_FONT, LABEL_FONT, LEGEND_FONT, COLORS, POLICY_LABELS
from privddnn.analysis.energy.extract_operation_energy import get_energy


matplotlib.rc('pdf', fonttype=42)
plt.rcParams['pdf.fonttype'] = 42


SERIES = ['', '_dnn_0', '_dnn_1', '_dnn_2']
SERIES_LABELS = ['Policy Alone', 'Exit 0', 'Exit 1', 'Exit 2']
ADJUSTMENT = 2


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trace-folder', type=str, required=True)
    parser.add_argument('--policies', type=str, nargs='+', required=True, choices=['random', 'max_prob', 'label_max_prob', 'cgr_max_prob'])
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.5))

        xs = np.arange(len(args.policies))
        width = 0.2
        offset = -width * (len(args.policies) - 1) / 2.0

        for policy_idx, policy in enumerate(args.policies):
            energy_list: List[float] = []
            std_list: List[float] = []

            for idx, series in enumerate(SERIES):
                meta_path = os.path.join(args.trace_folder, policy, '{}{}.json'.format(policy, series))
                metadata = read_json(meta_path)
                ops_per_trial, num_trials = metadata['ops_per_trial'], metadata['num_trials']

                trace_path = os.path.join(args.trace_folder, policy, '{}{}.csv'.format(policy, series))
                energy_results = get_energy(path=trace_path, num_trials=num_trials, should_plot=False)

                energy_per_op = [energy / ops_per_trial for energy in energy_results]
                avg_energy = np.average(energy_per_op)
                std_energy = np.std(energy_per_op)

                energy_list.append(avg_energy)
                std_list.append(std_energy)

                #annotation = '{:.3f}'.format(avg_energy) if avg_energy >= 1e-3 else '<0.001'
                #ax.annotate(annotation, (xs[idx] + offset, avg_energy), xytext=(xs[idx] + offset + xoffset, avg_energy + yoffset), fontsize=10)

            ax.bar(xs + offset, energy_list, width=width, label=POLICY_LABELS[policy], color=COLORS[policy], edgecolor='black', linewidth=1)
            ax.errorbar(xs + offset, energy_list, yerr=std_list, color='black', capsize=3, ls='none')
            offset += width

        ax.set_xticks(xs)
        ax.set_xticklabels(SERIES_LABELS, fontsize=LABEL_FONT + ADJUSTMENT + 1)

        ax.tick_params(axis='y', labelsize=LABEL_FONT + ADJUSTMENT + 1)

        ax.legend(fontsize=LEGEND_FONT + ADJUSTMENT + 1)
        ax.set_xlabel('Operation', size=AXIS_FONT + ADJUSTMENT - 0.5)
        ax.set_ylabel('Avg Energy (mJ)', size=AXIS_FONT + ADJUSTMENT + 2)
        ax.set_title('Average Energy per Operation', size=TITLE_FONT + ADJUSTMENT)

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
