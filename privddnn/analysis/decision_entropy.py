import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Tuple, DefaultDict

from privddnn.utils.plotting import PLOT_STYLE, MARKER_SIZE, LINE_WIDTH, COLORS
from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.metrics import compute_entropy


def seq_to_decimal(exit_decisions: List[int]) -> int:
    factor = 1
    conversion = 0

    for d in exit_decisions:
        conversion += factor * d
        factor *= 2

    return conversion


def compute_decision_entropy(output_levels: List[int], window_size: int) -> float:
    total_count = 0
    decision_counts = np.zeros(shape=(1 << window_size, ))

    for idx in range(len(output_levels) - window_size):
        key = seq_to_decimal(output_levels[idx:idx+window_size])
        decision_counts[key] += 1
        total_count += 1

    decision_probs = decision_counts / total_count
    return float(compute_entropy(probs=decision_probs, axis=0))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--window-size', type=int, required=True)
    args = parser.parse_args()

    test_log = read_json_gz(args.test_log)['test']

    # Compute the decision entropy in the purely random policy
    rand_entropy: List[float] = []
    for rate, results in sorted(test_log['random'].items()):
        entropy = compute_decision_entropy(output_levels=results[args.dataset_order]['output_levels'], window_size=args.window_size)
        rand_entropy.append(entropy)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        for policy_name in test_log.keys():
            entropy_results: List[float] = []
            exit_rates: List[float] = []
            normalized_entropy: List[float] = []

            for idx, (rate, results) in enumerate(sorted(test_log[policy_name].items())):
                entropy = compute_decision_entropy(output_levels=results[args.dataset_order]['output_levels'], window_size=args.window_size)
                entropy_results.append(entropy)
                exit_rates.append(rate)

                if rand_entropy[idx] > 1e-5:
                    midpoint = (entropy + rand_entropy[idx]) / 2.0
                    normalized_entropy.append((rand_entropy[idx] - entropy) / midpoint)

            ax.plot(exit_rates, entropy_results, label=policy_name, marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=COLORS[policy_name])

            avg_normalized_entropy = np.average(normalized_entropy)
            print('{} & {:.4f}'.format(policy_name, avg_normalized_entropy))

        ax.legend()
        ax.set_xlabel('Exit Rate')
        ax.set_ylabel('Exit Decision Sequence Entropy')
        ax.set_title('Entropy in Exit Decisions over Length-{} Windows'.format(args.window_size))

        plt.show()

        #print('{} & {}'.format(policy_name, ' & '.join(map(lambda s: '{:.5f}'.format(s), entropy_results))))

