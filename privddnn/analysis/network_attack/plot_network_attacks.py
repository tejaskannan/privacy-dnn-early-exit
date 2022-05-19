import numpy as np
import os.path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List

from privddnn.utils.file_utils import iterate_dir
from privddnn.utils.plotting import PLOT_STYLE, TITLE_FONT, AXIS_FONT, LEGEND_FONT, LABEL_FONT, POLICY_LABELS, COLORS
from privddnn.analysis.network_attack.evaluate_attack import evaluate_attack


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--results-folder', type=str, required=True)
    parser.add_argument('--attack-model-folder', type=str, required=True)
    parser.add_argument('--policies', type=str, nargs='+', required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    policy_names: List[str] = []
    exit_accuracies: List[float] = []
    attack_accuracies: List[float] = []

    for policy in sorted(args.policies):
        recovered_path = os.path.join(args.results_folder, policy, '{}_1000_recovered.jsonl.gz'.format(policy))
        true_path = os.path.join(args.results_folder, policy, '{}_1000.jsonl.gz'.format(policy))
        attack_path = [path for path in iterate_dir(args.attack_model_folder) if os.path.basename(path).startswith(policy)][0]

        exit_accuracy, attack_accuracy, _ = evaluate_attack(recovered_file=recovered_path,
                                                            true_file=true_path,
                                                            attack_model_file=attack_path)

        policy_names.append(policy)
        exit_accuracies.append(exit_accuracy * 100.0)
        attack_accuracies.append(attack_accuracy * 100.0)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        xs = np.arange(2)
        width = 0.2
        offset = -width * (len(policy_names) - 1) / 2

        for policy, exit_accuracy, attack_accuracy in zip(policy_names, exit_accuracies, attack_accuracies):
            ax.bar(xs + offset, [exit_accuracy, attack_accuracy], width=width, label=POLICY_LABELS[policy], edgecolor='black', linewidth=1, color=COLORS[policy])
            ax.annotate('{:.2f}'.format(exit_accuracy), (offset, exit_accuracy), (offset - 0.08, exit_accuracy + 1), fontsize=LABEL_FONT)
            ax.annotate('{:.2f}'.format(attack_accuracy), (1 + offset, attack_accuracy), (1 + offset - 0.08, attack_accuracy + 1), fontsize=LABEL_FONT)

            offset += width

        ax.legend(fontsize=LEGEND_FONT)

        ax.set_xticks(xs)
        ax.set_xticklabels(['Exit Recovery', 'Attack Accuracy'], fontsize=AXIS_FONT)
        
        ax.set_ylabel('Accuracy (%)', fontsize=AXIS_FONT)
        ax.set_title('Attacker Results from Network Traces', fontsize=TITLE_FONT)

        plt.show()
