import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from typing import DefaultDict, List

from privddnn.analysis.utils.read_logs import get_attack_results
from privddnn.utils.plotting import PLOT_STYLE, TITLE_FONT, AXIS_FONT, LEGEND_FONT, LINE_WIDTH, MARKER, MARKER_SIZE, POLICY_LABELS, COLORS


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log-folder', type=str, required=True)
    parser.add_argument('--dataset-orders', type=str, required=True, nargs='+')
    parser.add_argument('--attack-model', type=str, required=True)
    parser.add_argument('--policies', type=str, required=True, nargs='+')
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    policy_results: DefaultDict[str, List[float]] = defaultdict(list)
    window_sizes: List[int] = []

    for dataset_order in args.dataset_orders:
        attack_results = get_attack_results(folder_path=args.test_log_folder,
                                            fold='test',
                                            dataset_order=dataset_order,
                                            attack_train_log=args.test_log_folder,
                                            metric='accuracy',
                                            attack_model=args.attack_model,
                                            attack_policy='same',
                                            trials=1)

        window_sizes.append(int(dataset_order.split('-')[-1]))

        for policy, policy_attack_results in attack_results.items():
            max_accuracy = max([max(r) for r in policy_attack_results.values()])
            policy_results[policy].append(max_accuracy)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        for policy in args.policies:
            accuracy_values = policy_results[policy]
            ax.plot(window_sizes, accuracy_values, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=POLICY_LABELS[policy], color=COLORS[policy])

        ax.legend(fontsize=LEGEND_FONT)
        ax.set_xlabel('Block Size', fontsize=AXIS_FONT)
        ax.set_ylabel('Max Attack Accuracy (%)', fontsize=AXIS_FONT)

        ax.set_xticks(window_sizes)
        ax.set_xticklabels(window_sizes)

        ax.set_title('Maximum Attack Accuracy for Dataset Block Sizes', fontsize=TITLE_FONT)

        plt.tight_layout()

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
