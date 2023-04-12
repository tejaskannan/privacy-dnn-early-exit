import numpy as np
import os.path
import re
import time
from collections import defaultdict, Counter
from enum import Enum, auto
from typing import Dict, List, DefaultDict, Optional

from privddnn.utils.file_utils import read_json_gz, iterate_dir
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info
from privddnn.utils.ngrams import create_ngrams, create_ngram_counts
from privddnn.utils.inference_metrics import InferenceMetric, compute_metric
from privddnn.utils.constants import SMALL_NUMBER


TRIAL_REGEX = re.compile(r'.*trial([0-9]+).json.gz')
SUMMARY_TRIAL_REGEX = re.compile(r'.*trial([0-9]+)-summary.json.gz')
ATTACK_TRIAL_REGEX = re.compile(r'.*-attack-trial([0-9]+).json.gz')


def get_summary_results(folder_path: str,
                        fold: str,
                        dataset_order: str,
                        trials: Optional[int]) -> Dict[InferenceMetric, Dict[str, DefaultDict[str, List[float]]]]:
    """
    Gets the summary of the test results for each strategy and target rate(s)

    Args:
        folder_path: Path to the dataset test logs
        fold: The dataset fold to use (either 'val' or 'test')
        dataset_order: The dataset order to use (e.g. 'same-label-10')
        trials: An optional maximum number of trials to use
    Returns:
        A dictionary of { metric -> strategy -> { rate_str -> [ metric results for each trial ] } }
    """
    assert fold in ('val', 'test'), 'The fold must be either `val` or `test`'

    results: Dict[InferenceMetric, Dict[str, DefaultDict[str, List[float]]]] = dict()

    for log_path in iterate_dir(folder_path):
        match = SUMMARY_TRIAL_REGEX.match(log_path)
        if match is None:
            continue

        trial_num = int(match.group(1))
        if (trials is not None) and (trial_num >= trials):
            continue

        log_results = read_json_gz(log_path)[fold]

        for metric in InferenceMetric:
            if metric == InferenceMetric.COUNT_NGRAM_MI:
                continue

            if metric not in results:
                results[metric] = dict()

            file_name = os.path.split(log_path)[-1]
            strategy_name = file_name.split('-')[0]
            if strategy_name not in results[metric]:
                results[metric][strategy_name] = defaultdict(list)

            for rate_str, rate_results in log_results.items():
                results[metric][strategy_name][rate_str].append(rate_results[dataset_order][metric.name.lower()])

    return results


def get_test_results(folder_path: str,
                     fold: str,
                     dataset_order: str,
                     window_size: int,
                     trials: Optional[int]) -> Dict[InferenceMetric, Dict[str, DefaultDict[str, List[float]]]]:
    """
    Gets the test results for each strategy and target rate(s)

    Args:
        folder_path: Path to the dataset test logs
        fold: The dataset fold to use (either 'val' or 'test')
        dataset_order: The dataset order to use (e.g. 'same-label-10')
        window_size: The window size to use for Ngram mutual information
        trials: An optional maximum number of trials to use
    Returns:
        A dictionary of { metric -> strategy -> { rate_str -> [ metric results for each trial ] } }
    """
    assert fold in ('val', 'test'), 'The fold must be either `val` or `test`'

    results: Dict[InferenceMetric, Dict[str, DefaultDict[str, List[float]]]] = dict()

    for log_path in iterate_dir(folder_path):
        if 'attack' in log_path:
            continue

        log_results = read_json_gz(log_path)[fold]

        match = TRIAL_REGEX.match(log_path)
        trial_num = int(match.group(1))

        if (trials is not None) and (trial_num >= trials):
            continue

        for metric in InferenceMetric:
            if metric not in results:
                results[metric] = dict()

            file_name = os.path.split(log_path)[-1]
            strategy_name = file_name.split('-')[0]
            if strategy_name not in results[metric]:
                results[metric][strategy_name] = defaultdict(list)

            for rate_str, rate_results in log_results.items():
                preds = np.array(rate_results[dataset_order]['preds'])
                exit_decisions = np.array(rate_results[dataset_order]['output_levels'])
                labels = np.array(rate_results[dataset_order]['labels'])
                num_outputs = max(len(rate_str.split()), 2)

                metric_value = compute_metric(preds=preds, exit_decisions=exit_decisions, labels=labels, metric=metric, num_outputs=num_outputs)
                results[metric][strategy_name][rate_str].append(metric_value)

    return results


def get_attack_results(folder_path: str, fold: str, dataset_order: str, attack_train_log: str, attack_policy: str, metric: str, attack_model: str, target_pred: Optional[int], trials: Optional[int]) -> Dict[str, Dict[str, List[float]]]:
    """
    Gets the attack results for the logs in the given folder.

    Args:
        folder_path: Path to the dataset test logs
        fold: The dataset fold to use (either 'train' or 'test')
        dataset_order: The dataset order to use (e.g. 'same-label-10')
        attack_train_log: The folder for the testing logs used during attack training
        attack_policy: The name of the attack policy
        metric: The metric to calculate
        target_pred: The specific prediction to target. If not None, the metric must be 'accuracy'
        trial: Optional maximum number of trials
    Returns:
        A dictionary of strategy -> { rate_str -> metric result }
    """
    assert fold in ('train', 'test'), 'Attack fold must be either `train` or `test`'
    assert (target_pred is None) or (metric == 'accuracy'), 'If providing a target label, the accuracy must be `None`'

    # Make the attack key
    tokens = attack_train_log.split(os.sep)
    attack_train_file_name = tokens[-1] if len(tokens[-1]) > 0 else tokens[-2]
    attack_train_name = attack_train_file_name.replace('_test-logs', '')

    attack_key = '{}_{}'.format(attack_train_name, attack_policy)
    fold_name = 'attack_{}'.format(fold)

    results: Dict[str, Dict[str, List[float]]] = dict()

    if trials is not None:
        num_trials = trials
    else:
        num_trials = 0
        for log_path in iterate_dir(folder_path):
            match = ATTACK_TRIAL_REGEX.match(log_path)
            if match is None:
                continue

            trial = int(match.group(1))
            num_trials = max(num_trials, trial)

        num_trials += 1

    for log_path in iterate_dir(folder_path):
        match = ATTACK_TRIAL_REGEX.match(log_path)
        if match is None:
            continue

        trial = int(match.group(1))
        if trial >= num_trials:
            continue

        log = read_json_gz(log_path)
        if attack_key not in log:
            print('Could not find key {} in {}'.format(attack_key, log_path))
            continue

        log_results = log[attack_key][dataset_order][fold_name]

        file_name = os.path.split(log_path)[-1]
        strategy_name = file_name.split('-')[0]
        if strategy_name not in results:
            results[strategy_name] = dict()

        for rate, rate_results in log_results[attack_model].items():
            if rate not in results[strategy_name]:
                results[strategy_name][rate] = []

            if target_pred is not None:
                confusion_mat = rate_results['confusion_matrix']
                pred_results = [row[target_pred] for row in confusion_mat]
                result_value = float(pred_results[target_pred]) / float(np.sum(pred_results) + SMALL_NUMBER)
            else:
                result_value = rate_results[metric]

            if metric != 'correct_rank':
                result_value *= 100.0

            results[strategy_name][rate].append(result_value)

    return results
