import os.path
import scipy.stats as stats
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import Counter
from typing import Any, Dict, List

from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.plotting import PLOT_STYLE, AXIS_FONT, TITLE_FONT, LABEL_FONT


def get_exit_rates(log_results: Dict[str, Any]) -> List[float]:
    exit_counts: Counter = Counter()
    total_counts: Counter = Counter()

    for pred, level in zip(log_results['preds'], log_results['output_levels']):
        exit_counts[pred] += level
        total_counts[pred] += 1

    result: List[float] = []
    for pred in sorted(exit_counts.keys()):
        result.append(1.0 - (exit_counts[pred] / total_counts[pred]))

    return result


def get_series_label(log_path: str) -> str:
    dataset_name = log_path.split(os.sep)[-3]
    model_name = os.path.basename(log_path).split('_')[0]

    dataset_tokens = [t.upper() for t in dataset_name.split('_')]
    model_tokens = [t.upper() for t in model_name.split('-')]
    return ' '.join(dataset_tokens + model_tokens)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--log1', type=str, required=True)
    parser.add_argument('--log2', type=str, required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--exit-rate', type=float, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    log1 = read_json_gz(args.log1)['val'][args.policy]
    log2 = read_json_gz(args.log2)['test'][args.policy]

    exit_rate = str(round(float(args.exit_rate), 2))
    log1_results = log1[exit_rate][args.dataset_order]
    log2_results = log2[exit_rate][args.dataset_order]

    log1_exit_rates: List[float] = get_exit_rates(log1_results)
    log2_exit_rates: List[float] = get_exit_rates(log2_results)

    r, pval = stats.pearsonr(log1_exit_rates, log2_exit_rates)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        # Plot the points
        ax.scatter(log1_exit_rates, log2_exit_rates)

        # List the correlation
        ax.annotate('r: {:.4f}'.format(r), xy=(0.7, 0.05), fontsize=LABEL_FONT)

        policy_label = ''.join(t.capitalize() for t in args.policy.split('_'))
        ax.set_title('Per-Prediction {} Exit Rates for Target {}'.format(policy_label, exit_rate), fontsize=TITLE_FONT)

        ax.tick_params(axis='x', labelsize=LABEL_FONT)
        ax.tick_params(axis='y', labelsize=LABEL_FONT)

        ax.set_xlabel(get_series_label(args.log1), fontsize=AXIS_FONT)
        ax.set_ylabel(get_series_label(args.log2), fontsize=AXIS_FONT)

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, transparent=True, bbox_inches='tight')
