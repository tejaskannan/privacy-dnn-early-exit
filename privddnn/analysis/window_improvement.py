import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Dict

from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info
from privddnn.utils.plotting import PLOT_STYLE, AXIS_FONT, LEGEND_FONT, TITLE_FONT, DATASET_LABELS
from improvement_area import area_between_curves


COLORS = {
    10: '#a1dab4',
    20: '#41bc64',
    30: '#2c7fb8',
    40: '#253494'
}

METRIC_NAMES = {
    'accuracy': 'Accuracy',
    'mutual_information': 'Mutual Information'
}


def get_dataset_name(test_log_path: str) -> str:
    path_tokens = test_log_path.split(os.sep)
    saved_models_idx = path_tokens.index('saved_models')
    return path_tokens[saved_models_idx + 1]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-logs', type=str, required=True, nargs='+')
    parser.add_argument('--dataset-order', type=str, required=True, choices=['nearest', 'same-label'])
    parser.add_argument('--metric', type=str, required=True, choices=['accuracy', 'mutual_information'])
    parser.add_argument('--window-sizes', type=int, required=True, nargs='+')
    args = parser.parse_args()

    # Get the data-dependent accuracy for each window size
    baseline_policies = ['max_prob', 'adaptive_random_max_prob', 'random']
    metrics: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = dict()  # Dataset -> { window -> { policy -> { rate -> accuracy }}}

    for test_log_path in args.test_logs:
        test_log = read_json_gz(test_log_path)['test']

        dataset_name = get_dataset_name(test_log_path)
        dataset_metrics: Dict[int, Dict[str, Dict[str, float]]] = dict()

        for window_size in args.window_sizes:
            policy_metrics: Dict[str, Dict[str, float]] = dict()
            dataset_order = '{}-{}'.format(args.dataset_order, window_size)

            for policy_name in baseline_policies:
                window_metrics: Dict[str, float] = dict()
                rates: List[str] = []

                for rate, results in test_log[policy_name].items():
                    preds = np.array(results[dataset_order]['preds'])
                    exit_decisions = np.array(results[dataset_order]['output_levels'])
                    labels = np.array(results[dataset_order]['labels'])

                    if args.metric == 'accuracy':
                        window_metrics[rate] = compute_accuracy(predictions=preds, labels=labels)
                    elif args.metric == 'mutual_information':
                        window_metrics[rate] = compute_mutual_info(X=exit_decisions, Y=preds, should_normalize=False)

                    rates.append(rate)

                policy_metrics[policy_name] = window_metrics

            dataset_metrics[window_size] = policy_metrics

        metrics[dataset_name] = dataset_metrics

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        width = 0.25
        offset = -width * ((len(args.window_sizes) - 1) / 2)
        xs = np.arange(len(metrics))

        for window_size in args.window_sizes:
            improvement_scores: List[float] = []

            for dataset_name, dataset_metrics in sorted(metrics.items()):
                data_dependent_area = area_between_curves(rates=rates,
                                                          upper=dataset_metrics[window_size]['max_prob'],
                                                          lower=dataset_metrics[window_size]['random'])
                policy_area = area_between_curves(rates=rates,
                                                  upper=dataset_metrics[window_size]['adaptive_random_max_prob'],
                                                  lower=dataset_metrics[window_size]['random'])

                improvement_score = policy_area / data_dependent_area

                if args.metric == 'mutual_information':
                    improvement_score = 1.0 - improvement_score

                improvement_scores.append(improvement_score)

            for x, score in zip(xs, improvement_scores):
                x += offset
                ax.annotate('{:.4f}'.format(score), xy=(x, score), xytext=(x - 0.13, score + 0.002))

            ax.bar(xs + offset, improvement_scores, width=width, label='Window {}'.format(window_size), color=COLORS[window_size])
            offset += width

        ax.legend(fontsize=LEGEND_FONT)

        ax.set_xticks(xs)
        ax.set_xticklabels([DATASET_LABELS[dataset] for dataset in sorted(metrics.keys())])

        metric_name = METRIC_NAMES[args.metric]

        ax.set_xlabel('Dataset', size=AXIS_FONT)
        ax.set_ylabel('{} Improvement Score'.format(metric_name), size=AXIS_FONT)
        ax.set_title('{} Improvement for Varying Window Sizes'.format(metric_name), size=TITLE_FONT)

        plt.show()
