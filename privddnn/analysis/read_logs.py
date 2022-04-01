import numpy as np
import os.path
import re
from collections import defaultdict, Counter
from typing import Dict, List, DefaultDict, Optional

from privddnn.utils.file_utils import read_json_gz, iterate_dir
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info
from privddnn.utils.ngrams import create_ngrams, create_ngram_counts


TRIAL_REGEX = re.compile(r'.*trial([0-9]+).json.gz')
ATTACK_TRIAL_REGEX = re.compile(r'.*-attack-trial([0-9]+).json.gz')
INPUT_ATTACK_TRIAL_REGEX = re.compile(r'.*-attack_input-trial([0-9]+).json.gz')


def get_test_results(folder_path: str, fold: str, dataset_order: str, metric: str, trials: Optional[int]) -> Dict[str, DefaultDict[str, List[float]]]:
    """
    Gets the test results for each strategy and target rate(s)

    Args:
        folder_path: Path to the dataset test logs
        fold: The dataset fold to use (either 'val' or 'test')
        dataset_order: The dataset order to use (e.g. 'same-label-10')
        metric: The metric to calculate
        trials: An optional maximum number of trials to use
    Returns:
        A dictionary of strategy -> { rate_str -> [ metric results for each trial ] }
    """
    assert fold in ('val', 'test'), 'The fold must be either `val` or `test`'

    results: Dict[str, DefaultDict[str, List[float]]] = dict()

    for log_path in iterate_dir(folder_path):
        if 'attack' in log_path:
            continue

        log_results = read_json_gz(log_path)[fold]

        match = TRIAL_REGEX.match(log_path)
        trial_num = int(match.group(1))

        if (trials is not None) and (trial_num >= trials):
            continue

        file_name = os.path.split(log_path)[-1]
        strategy_name = file_name.split('-')[0]
        if strategy_name not in results:
            results[strategy_name] = defaultdict(list)

        for rate_str, rate_results in log_results.items():
            preds = np.array(rate_results[dataset_order]['preds'])
            exit_decisions = np.array(rate_results[dataset_order]['output_levels'])
            labels = np.array(rate_results[dataset_order]['labels'])

            if metric == 'accuracy':
                accuracy = compute_accuracy(predictions=preds, labels=labels)
                results[strategy_name][rate_str].append(accuracy)
            elif metric == 'mutual_information':
                mutual_information = compute_mutual_info(X=exit_decisions, Y=preds)
                results[strategy_name][rate_str].append(mutual_information)
            elif metric == 'exit_rate_deviation':
                exit_counts: Counter = Counter()
                pred_counts: Counter = Counter()

                for pred, decision in zip(preds, exit_decisions):
                    exit_counts[pred] += decision
                    pred_counts[pred] += 1

                avg_exits_list: List[float] = []
                for label in range(np.amax(labels) + 1):
                    avg_exit = exit_counts[label] / max(pred_counts[label], 1)
                    avg_exits_list.append(avg_exit)

                exit_deviation = np.std(avg_exits_list)
                results[strategy_name][rate_str].append(exit_deviation)
            elif metric.startswith('ngram'):
                ngram_size = int(metric.split('_')[-1])
                num_outputs = max(len(rate_str.split()), 2)
                ngram_decisions, ngram_preds = create_ngrams(levels=exit_decisions,
                                                             preds=preds,
                                                             n=ngram_size,
                                                             num_outputs=num_outputs)
                ngram_mi = compute_mutual_info(X=ngram_decisions, Y=ngram_preds)
                results[strategy_name][rate_str].append(ngram_mi)
            elif metric.startswith('count_ngram'):
                ngram_size = int(metric.split('_')[-1])
                ngram_decisions, ngram_preds = create_ngram_counts(levels=exit_decisions, preds=preds, n=ngram_size)
                ngram_mi = compute_mutual_info(X=ngram_decisions, Y=ngram_preds)
                results[strategy_name][rate_str].append(ngram_mi)
            else:
                raise ValueError('Unknown metric name: {}'.format(metric))

    return results


def get_attack_results(folder_path: str, fold: str, dataset_order: str, attack_train_log: str, attack_policy: str, metric: str, attack_model: str, trials: Optional[int]) -> Dict[str, Dict[str, List[float]]]:
    """
    Gets the attack results for the logs in the given folder.

    Args:
        folder_path: Path to the dataset test logs
        fold: The dataset fold to use (either 'train' or 'test')
        dataset_order: The dataset order to use (e.g. 'same-label-10')
        attack_train_log: The folder for the testing logs used during attack training
        attack_policy: The name of the attack policy
        metric: The metric to calculate
        trial: Optional maximum number of trials
    Returns:
        A dictionary of strategy -> { rate_str -> metric result }
    """
    assert fold in ('train', 'test'), 'Attack fold must be either `train` or `test`'

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

        log_results = read_json_gz(log_path)[attack_key][dataset_order][fold_name]

        file_name = os.path.split(log_path)[-1]
        strategy_name = file_name.split('-')[0]
        if strategy_name not in results:
            results[strategy_name] = dict()

        if attack_model == 'best':
            for model_name, model_results in log_results.items():
                for rate, rate_results in log_results[model_name].items():
                    if rate not in results[strategy_name]:
                        results[strategy_name][rate] = [0.0 for _ in range(num_trials + 1)]

                    results[strategy_name][rate][trial] = max(results[strategy_name][rate][trial], rate_results[metric])
        else:
            for rate, rate_results in log_results[attack_model].items():
                if rate not in results[strategy_name]:
                    results[strategy_name][rate] = []

                results[strategy_name][rate].append(rate_results[metric])

    return results


def get_input_attack_results(folder_path: str, fold: str, dataset_order: str, metric: str, attack_model: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Gets the attack results for the logs in the given folder.

    Args:
        folder_path: Path to the dataset test logs
        fold: The dataset fold to use (either 'train' or 'test')
        dataset_order: The dataset order to use (e.g. 'same-label-10')
        metric: The metric to calculate
        attack_model: The attack model name to use
    Returns:
        A dictionary of strategy -> { rate_str -> [metric result per trial] }
    """
    assert fold in ('train', 'test'), 'Attack fold must be either `train` or `test`'

    # Make the attack key
    results: Dict[str, Dict[str, List[float]]] = dict()

    num_trials = 0
    for log_path in iterate_dir(folder_path):
        match = INPUT_ATTACK_TRIAL_REGEX.match(log_path)
        if match is None:
            continue

        trial = int(match.group(1))
        num_trials = max(num_trials, trial)

    for log_path in iterate_dir(folder_path):
        match = INPUT_ATTACK_TRIAL_REGEX.match(log_path)
        if match is None:
            continue

        log_results = read_json_gz(log_path)[dataset_order][fold]
        trial = int(match.group(1))

        file_name = os.path.split(log_path)[-1]
        strategy_name = file_name.split('-')[0]
        if strategy_name not in results:
            results[strategy_name] = dict()

        for rate, rate_results in log_results[attack_model].items():
            if rate not in results[strategy_name]:
                results[strategy_name][rate] = []

            results[strategy_name][rate].append(rate_results[metric])

    return results
