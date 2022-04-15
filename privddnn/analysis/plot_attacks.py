import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Dict, Set

from privddnn.analysis.read_logs import get_attack_results
from privddnn.utils.constants import BIG_NUMBER
from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info
from privddnn.utils.plotting import PLOT_STYLE, AXIS_FONT, LEGEND_FONT, TITLE_FONT, DATASET_LABELS, COLORS, to_label

METRIC_NAMES = {
    'accuracy': 'Accuracy',
    'top2': 'Top 2 Accuracy',
    'correct_rank': 'ADCS'
}


def get_dataset_name(test_log_path: str) -> str:
    path_tokens = test_log_path.split(os.sep)
    saved_models_idx = path_tokens.index('saved_models')
    return path_tokens[saved_models_idx + 1]


def get_max_metric(metrics: Dict[str, List[float]]) -> float:
    max_val = -BIG_NUMBER
    for series_values in metrics.values():
        max_val = max(max_val, np.average(series_values))

    return max_val


def get_min_metric(metrics: Dict[str, List[float]]) -> float:
    min_val = BIG_NUMBER
    for series_values in metrics.values():
        min_val = min(min_val, np.average(series_values))

    return min_val


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log-folders', type=str, required=True, nargs='+')
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True, choices=['accuracy', 'top2', 'correct_rank'])
    parser.add_argument('--trials', type=int, required=True)
    parser.add_argument('--should-normalize', action='store_true')
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    # Get the data-dependent accuracy for each window size
    metrics: Dict[str, Dict[str, Dict[str, List[float]]]] = dict()  # Dataset -> { policy -> { rate -> [metric] }}
    policy_names_set: Set[str] = set()

    for test_log_folder in args.test_log_folders:
        dataset_results = get_attack_results(folder_path=test_log_folder,
                                             fold='test',
                                             dataset_order=args.dataset_order,
                                             attack_train_log=test_log_folder,
                                             metric=args.metric,
                                             attack_policy='same',
                                             attack_model='Majority',
                                             trials=args.trials)

        path_tokens = test_log_folder.split(os.sep)
        dataset_name = path_tokens[-3] if len(path_tokens[-1]) > 0 else path_tokens[-4]

        metrics[dataset_name] = dataset_results
        policy_names_set.update(dataset_results.keys())

    # Standardize the order of all datasets
    dataset_names = list(sorted(metrics.keys()))
    policy_names = list(sorted(policy_names_set))

    # Aggregate the metrics per dataset
    aggregate_metrics: Dict[str, List[float]] = dict()  # Policy -> [ dataset metrics ]
    for policy_name in policy_names:
        aggregate_metrics[policy_name] = []

        for dataset_name in dataset_names:
            rate_results = metrics[dataset_name][policy_name]

            if args.metric == 'correct_rank':
                aggregate = get_min_metric(rate_results)
            else:
                aggregate = get_max_metric(rate_results) * 100.0

            aggregate_metrics[policy_name].append(aggregate)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(14, 4))

        width = 0.2
        offset = -width * (len(policy_names) - 1) / 2
        xs = np.arange(len(dataset_names))

        for policy_name in policy_names:
            policy_results = aggregate_metrics[policy_name]

            if args.should_normalize:
                policy_results = [(res / base) for res, base in zip(policy_results, aggregate_metrics['random'])]

            ax.bar(xs + offset, policy_results, width=width, label=to_label(policy_name), color=COLORS[policy_name])
            
            for x, metric in zip(xs + offset, policy_results):
                ax.annotate('{:.2f}'.format(metric), xy=(x, metric), xytext=(x - 0.1, metric * 1.01))

            offset += width

        if not args.should_normalize:
            ax.legend(fontsize=LEGEND_FONT)

        ax.set_xticks(xs)
        ax.set_xticklabels([DATASET_LABELS[dataset] for dataset in sorted(metrics.keys())])

        metric_name = METRIC_NAMES[args.metric]
        aggregate_name = 'Min' if args.metric == 'correct_rank' else 'Max'

        ylabel = '{} {}'.format(aggregate_name, metric_name)

        if args.should_normalize:
            ylabel = 'Normalized {}'.format(ylabel)

        ax.set_xlabel('Dataset', size=AXIS_FONT)
        ax.set_ylabel(ylabel, size=AXIS_FONT)
        ax.set_title('Attacker {} {} For Each Task'.format(aggregate_name, metric_name), size=TITLE_FONT)

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, transparent=True, bbox_inches='tight')
