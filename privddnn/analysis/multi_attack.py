import os
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from typing import List

from privddnn.attack.attack_classifiers import MOST_FREQ, MAJORITY, LOGISTIC_REGRESSION_COUNT, NGRAM
from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.plotting import to_label, COLORS, AXIS_FONT, TITLE_FONT, LEGEND_FONT
from privddnn.utils.plotting import LINE_WIDTH, MARKER, MARKER_SIZE, PLOT_STYLE
from privddnn.dataset.dataset import Dataset

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True, choices=['accuracy', 'top2', 'top5', 'top10', 'top(k-1)', 'weighted_accuracy'])
    parser.add_argument('--attack-model', type=str, required=True, choices=[MAJORITY, LOGISTIC_REGRESSION_COUNT, MOST_FREQ, NGRAM, 'best'])
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--attack-train-log', type=str)
    parser.add_argument('--train-policy', type=str, default='same')
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    # Read the attack accuracy
    test_log = read_json_gz(args.test_log)

    attack_train_log = args.test_log if args.attack_train_log is None else args.attack_train_log
    attack_train_log_name = os.path.basename(attack_train_log)

    attack_key = '{}_{}'.format(attack_train_log_name.replace('_test-log.json.gz', ''), args.train_policy)
    attack_accuracy = test_log[attack_key][args.dataset_order]['attack_test']

    for policy_name, attack_results in attack_accuracy.items():

        if args.attack_model == 'best':
            all_results: List[List[float]] = []
            for attack_result in attack_results.values():
                all_results.append([r[args.metric] * 100.0 for r in attack_result])

            metric_results = np.max(all_results, axis=0)
        else:
            metric_results = [r[args.metric] * 100.0 for r in attack_results[args.attack_model]]

        average_metric = sum(metric_results) / len(metric_results)
        med_metric = np.median(metric_results)
        std_metric = np.std(metric_results)
        iqr_metric = np.percentile(metric_results, 75) - np.percentile(metric_results, 25)
        max_metric = max(metric_results)

        print('{} & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f}'.format(policy_name, average_metric, std_metric, med_metric, iqr_metric, max_metric))
