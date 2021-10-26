import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import Counter, namedtuple
from typing import List, Dict

from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.plotting import PLOT_STYLE, to_label, COLORS, LEGEND_FONT, AXIS_FONT, TITLE_FONT


ExitTuple = namedtuple('ExitTuple', ['observed', 'lower', 'upper'])


def get_exit_counts(preds: List[int], output_levels: List[int], target_pred: int, rate: float, window: int) -> ExitTuple:
    observed_list: List[int] = []
    lower_list: List[float] = []
    upper_list: List[float] = []

    cumulative_count = 0
    total_count = 0

    for pred, level in filter(lambda p: (p[0] == target_pred), zip(preds, output_levels)):
        cumulative_count += level
        total_count += 1

        expected = (1.0 - rate) * total_count
        lower = expected - window
        upper = expected + window

        observed_list.append(cumulative_count)
        lower_list.append(lower)
        upper_list.append(upper)

    return ExitTuple(observed=observed_list, lower=lower_list, upper=upper_list)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--target-pred', type=int, required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--policy-name', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    test_log = read_json_gz(args.test_log)['test']
    policies = list(test_log.keys())

    rate_results = test_log[args.policy_name][str(round(args.rate, 2))]
    preds = rate_results[0]['preds']
    output_levels = rate_results[0]['output_levels']

    exit_count_tuple = get_exit_counts(preds=preds,
                                       output_levels=output_levels,
                                       target_pred=args.target_pred,
                                       rate=args.rate,
                                       window=args.window)

    xs = np.arange(len(exit_count_tuple.observed))

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        ax.plot(xs, exit_count_tuple.observed, label='Observed')
        ax.plot(xs, exit_count_tuple.lower, label='Lower')
        ax.plot(xs, exit_count_tuple.upper, label='Upper')

        ax.legend(fontsize=LEGEND_FONT)

        ax.set_xlabel('Sample Count', fontsize=AXIS_FONT)
        ax.set_ylabel('Elevation Count', fontsize=AXIS_FONT)
        ax.set_title('Elevation Counts over Time', fontsize=TITLE_FONT)

        #ax.set_xticks(xs)
        #ax.set_xticklabels(xs, fontsize=LEGEND_FONT)

        #ax.set_yticks(list(map(lambda y: round(y, 1), ax.get_yticks())))
        #ax.set_yticklabels(ax.get_yticks(), fontsize=LEGEND_FONT)

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
