import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import Counter
from typing import List, Dict

from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.plotting import PLOT_STYLE, to_label, COLORS, LEGEND_FONT, AXIS_FONT, TITLE_FONT


def get_exit_rate_per_label(preds: List[int], output_levels: List[int]) -> List[float]:
    elevate_counter: Counter = Counter()
    total_counter: Counter = Counter()

    for pred, level in zip(preds, output_levels):
        elevate_counter[pred] += level
        total_counter[pred] += 1

    result: List[float] = []

    for pred in sorted(elevate_counter.keys()):
        result.append(elevate_counter[pred] / total_counter[pred])

    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    test_log = read_json_gz(args.test_log)['test']
    policies = list(test_log.keys())

    exit_rates: Dict[str, List[float]] = dict()

    for policy_name in policies:
                
        rate_results = test_log[policy_name][str(round(args.rate, 2))]
        preds = rate_results[0]['preds']
        output_levels = rate_results[0]['output_levels']

        exit_rate = get_exit_rate_per_label(preds=preds, output_levels=output_levels)
        num_labels = np.amax(preds) + 1

        exit_rates[policy_name] = exit_rate

    xs = np.arange(num_labels)
    width = 1.0 / (len(policies) + 1)
    offset = (-1 * (len(policies) - 1) / 2) * width

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        for policy_name in policies:
            exit_rate = exit_rates[policy_name]

            ax.bar(xs + offset, exit_rate, width, label=to_label(policy_name), color=COLORS[policy_name])
            offset += width

        ax.legend(fontsize=LEGEND_FONT)

        ax.set_xlabel('Predicted Class', fontsize=AXIS_FONT)
        ax.set_ylabel('Elevation Rate', fontsize=AXIS_FONT)
        ax.set_title('Elevation Rate Stratified by Prediction', fontsize=TITLE_FONT)

        ax.set_xticks(xs)
        ax.set_xticklabels(xs, fontsize=LEGEND_FONT)

        ax.set_yticks(list(map(lambda y: round(y, 1), ax.get_yticks())))
        ax.set_yticklabels(ax.get_yticks(), fontsize=LEGEND_FONT)

        # Denote the expected amount using a dotted line
        ax.axhline(1.0 - args.rate, linestyle='--', color='k')

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
        
        #row = [policy_name] + list(map(lambda r: '{:.4f}'.format(r), exit_rate))
        #print(' & '.join(row) + '\\\\')

