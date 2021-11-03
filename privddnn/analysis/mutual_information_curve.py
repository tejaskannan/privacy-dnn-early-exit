import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List

from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.constants import SMALL_NUMBER
from privddnn.utils.metrics import compute_mutual_info
from privddnn.utils.plotting import MARKER, MARKER_SIZE, LINE_WIDTH, COLORS, to_label, PLOT_STYLE


def mut_info_upper_bound(pred_counts: List[int], hard_cap: int, rate: float, step: int):
    bound = 0.0

    for pred_count in pred_counts:
        expected_exit = rate * pred_count
        expected_stay = (1.0 - rate) * pred_count

        term0 = (pred_count / step) * (rate * np.log(1 + (hard_cap / expected_exit)))
        term1 = (pred_count / step) * ((1.0 - rate) * np.log(1 + (hard_cap / expected_stay)))
        term2 = (hard_cap / step) * np.abs(np.log((expected_exit + hard_cap) / max(expected_stay - hard_cap, SMALL_NUMBER)))

        bound += term0 + term1 + term2

    return bound


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--hard-cap', type=int, required=True)
    args = parser.parse_args()

    assert (args.rate >= 0.0) and (args.rate <= 1.0), 'Rate must be in [0, 1]'
    assert args.window > 0, 'Must provide a positive window'

    test_log = read_json_gz(args.test_log)['test']
    rate = str(round(args.rate, 2))
    window = args.window
    policies = list(test_log.keys())

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        for policy_name in policies:
            if policy_name != 'even_max_prob':
                continue

            output_levels = np.array(test_log[policy_name][rate][0]['output_levels'])
            preds = np.array(test_log[policy_name][rate][0]['preds'])
            num_samples = len(preds)

            num_samples_list: List[int] = []
            mut_info_list: List[float] = []

            for sample_count in range(window, num_samples, window):
                mi = compute_mutual_info(X=output_levels[0:sample_count],
                                         Y=preds[0:sample_count],
                                         should_normalize=False)

                num_samples_list.append(sample_count)
                mut_info_list.append(mi)

            ax.plot(num_samples_list, mut_info_list, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=to_label(policy_name), color=COLORS[policy_name])

            num_labels = np.max(preds) + 1
            upper_bound: List[int] = []
            upper_bound_xs: List[int] = []

            for sample_count in range(window, num_samples, window):
                pred_counts = np.bincount(preds[0:sample_count], minlength=num_labels)
                
                if (np.min(pred_counts) * (1.0 - args.rate) > args.hard_cap):
                    upper_bound.append(mut_info_upper_bound(pred_counts, hard_cap=args.hard_cap, rate=args.rate, step=sample_count))
                    upper_bound_xs.append(sample_count)

            ax.plot(upper_bound_xs, upper_bound, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='Upper Bound', color='red')

        plt.show()
