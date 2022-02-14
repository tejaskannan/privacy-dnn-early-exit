import numpy as np
import os.path
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum, auto
from typing import List, Tuple, DefaultDict, Dict, Optional

from privddnn.dataset import Dataset
from privddnn.dataset.data_iterators import make_data_iterator
from privddnn.classifier import BaseClassifier, ModelMode, OpName
from privddnn.ensemble.adaboost import AdaBoostClassifier
from privddnn.exiting.early_exit import ExitStrategy, EarlyExiter, make_policy, EarlyExitResult
from privddnn.restore import restore_classifier
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info, compute_target_exit_rates, compute_stop_counts
from privddnn.utils.plotting import to_label
from privddnn.utils.file_utils import save_json_gz, read_json_gz


def execute_fixed_strategy(clf: BaseClassifier,
                           dataset: Dataset,
                           exit_rate: float,
                           data_iterator_name: str,
                           window_size: Optional[int],
                           num_reps: int,
                           fold: str,
                           max_num_samples: Optional[int]) -> Dict[str, List[float]]:
    assert isinstance(clf, AdaBoostClassifier), 'The fixed strategy only works with AdaBoost.'

    # Collect the dataset
    data_iterator = make_data_iterator(name=data_iterator_name,
                                       dataset=dataset,
                                       clf=None,
                                       window_size=window_size,
                                       num_reps=num_reps,
                                       fold=fold)

    inputs_list: List[np.ndarray] = []
    labels_list: List[int] = []

    for input_features, _, label in data_iterator:
        inputs_list.append(np.expand_dims(input_features, axis=0))
        labels_list.append(int(label))

    inputs = np.vstack(inputs_list)

    # Execute the classifier for this rate
    probs = clf.predict_proba_for_rate(inputs=inputs, rate=exit_rate)
    preds = np.argmax(probs, axis=-1)

    return {
        data_iterator.name: {
            'preds': preds.astype(int).tolist(),
            'labels': labels_list,
            'output_levels': [0 for _ in labels_list],
            'window_size': window_size,
            'num_reps': num_reps
        }
    }


def execute_for_rate(dataset: Dataset,
                     val_probs: np.ndarray,
                     val_labels: np.ndarray,
                     test_probs: np.ndarray,
                     data_iterator_name: str,
                     window_size: Optional[int],
                     pred_rates: np.ndarray,
                     rate: float,
                     model_path: str,
                     strategy: ExitStrategy,
                     num_reps: int,
                     max_num_samples: Optional[int]) -> Dict[str, Dict[str, List[float]]]:
    # Make the exit policy
    rates = [rate, 1.0 - rate]
    policy = make_policy(strategy=strategy, rates=rates, model_path=model_path)
    policy.fit(val_probs=val_probs, val_labels=val_labels)

    # Run the policy on the validation and test sets
    val_iterator = make_data_iterator(name=data_iterator_name,
                                      dataset=dataset,
                                      pred_probs=val_probs,
                                      window_size=window_size,
                                      num_reps=num_reps,
                                      fold='val')
    val_result = policy.test(data_iterator=val_iterator,
                             num_labels=dataset.num_labels,
                             pred_rates=pred_rates,
                             max_num_samples=max_num_samples)

    test_iterator = make_data_iterator(name=data_iterator_name,
                                       dataset=dataset,
                                       pred_probs=test_probs,
                                       window_size=window_size,
                                       num_reps=num_reps,
                                       fold='test')
    test_result = policy.test(data_iterator=test_iterator,
                              num_labels=dataset.num_labels,
                              pred_rates=pred_rates,
                              max_num_samples=max_num_samples)

    result = dict(val=dict(), test=dict())

    val_dict = {
        'preds': val_result.predictions.tolist(),
        'output_levels': val_result.output_levels.tolist(),
        'labels': val_result.labels.tolist(),
        'num_changed': val_result.num_changed,
        'selection_counts': { key.name: value for key, value in val_result.selection_counts.items() },
        'window_size': window_size,
        'num_reps': num_reps
    }

    test_dict = {
        'preds': test_result.predictions.tolist(),
        'output_levels': test_result.output_levels.tolist(),
        'labels': test_result.labels.tolist(),
        'num_changed': test_result.num_changed,
        'selection_counts': { key.name: value for key, value in test_result.selection_counts.items() },
        'window_size': window_size,
        'num_reps': num_reps
    }

    result['val'][val_iterator.name] = val_dict
    result['test'][test_iterator.name] = test_dict

    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--dataset-order', type=str, required=True, help='The type of data iterator to create.')
    parser.add_argument('--reps', type=int, default=1, help='The number of repetitions of the dataset.')
    parser.add_argument('--window-size', type=int, help='The window size used to build the dataset.')
    parser.add_argument('--max-num-samples', type=int, help='Optional maximum number of samples (for testing)')
    args = parser.parse_args()

    assert args.reps >= 1, 'Must provide a positive number of dataset repititions'

    # Restore the model
    model: BaseClassifier = restore_classifier(model_path=args.model_path, model_mode=ModelMode.TEST)

    # Get the predictions from the models
    val_probs = model.validate()  # [B, L, K]
    test_probs = model.test()  # [C, L, K]

    # Get the validation labels (we use this to fit any policies)
    val_labels = model.dataset.get_val_labels()  # [B]
    stop_counts = compute_stop_counts(probs=val_probs)

    rates = list(sorted(np.arange(0.0, 1.01, 0.1)))
    rand = np.random.RandomState(seed=591)

    # Execute all early stopping policies
    #strategies = [ExitStrategy.ADAPTIVE_RANDOM_MAX_PROB, ExitStrategy.MAX_PROB, ExitStrategy.RANDOM]
    strategies = [ExitStrategy.MAX_PROB, ExitStrategy.RANDOM]

    # Load the existing test log (if present)
    file_name = os.path.basename(args.model_path).split('.')[0]
    test_log_name = '{}_test-log.json.gz'.format(file_name)
    test_log_path = os.path.join(os.path.dirname(args.model_path), test_log_name)

    results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = dict(val=dict(), test=dict())
    if os.path.exists(test_log_path):
        results = read_json_gz(test_log_path)

    for strategy in strategies:
        strategy_name = strategy.name.lower()

        if strategy_name not in results['val']:
            results['val'][strategy_name] = dict()

        if strategy_name not in results['test']:
            results['test'][strategy_name] = dict()

        for rate in reversed(rates):
            print('Testing {} on {:.2f}'.format(strategy_name.capitalize(), float(round(rate, 2))), end='\r')

            if strategy == ExitStrategy.FIXED:
                rate_result: Dict[str, Dict[str, Dict[str, List[float]]]] = dict()

                for fold in ['val', 'test']:
                    rate_result[fold] = execute_fixed_strategy(clf=model,
                                                               dataset=model.dataset,
                                                               exit_rate=rate,
                                                               data_iterator_name=args.dataset_order,
                                                               window_size=args.window_size,
                                                               num_reps=args.reps,
                                                               fold=fold,
                                                               max_num_samples=args.max_num_samples)
            else:
                rate_result = execute_for_rate(dataset=model.dataset,
                                               val_probs=val_probs,
                                               val_labels=val_labels,
                                               test_probs=test_probs,
                                               pred_rates=stop_counts,
                                               rate=rate,
                                               model_path=args.model_path,
                                               strategy=strategy,
                                               data_iterator_name=args.dataset_order,
                                               window_size=args.window_size,
                                               num_reps=args.reps,
                                               max_num_samples=args.max_num_samples)

            # Log the results
            rate_key = str(round(rate, 2))

            if rate_key not in results['val'][strategy_name]:
                results['val'][strategy_name][rate_key] = dict()

            if rate_key not in results['test'][strategy_name]:
                results['test'][strategy_name][rate_key] = dict()

            results['val'][strategy_name][rate_key].update(rate_result['val'])
            results['test'][strategy_name][rate_key].update(rate_result['test'])

        print()

    # Save the results into the test log
    file_name = os.path.basename(args.model_path).split('.')[0]
    test_log_name = '{}_test-log.json.gz'.format(file_name)
    test_log_path = os.path.join(os.path.dirname(args.model_path), test_log_name)

    save_json_gz(results, test_log_path)
