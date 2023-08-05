import numpy as np
import os.path
import time
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum, auto
from itertools import permutations
from typing import List, Tuple, DefaultDict, Dict, Optional, Any

from privddnn.dataset import Dataset
from privddnn.dataset.data_iterators import make_data_iterator
from privddnn.classifier import BaseClassifier, ModelMode, OpName
from privddnn.exiting import ExitStrategy, EarlyExiter, make_policy, EarlyExitResult
from privddnn.exiting import POLICY_HAS_RANDOMNESS
from privddnn.restore import restore_classifier
from privddnn.utils.file_utils import save_json_gz, make_dir, read_json_gz
from privddnn.utils.exit_utils import get_exit_rates
from privddnn.utils.inference_metrics import compute_metric, InferenceMetric


TARGET_BOUNDS = {
    2: (0.0, 1.0),
    3: (0.2, 0.7),
    4: (0.2, 0.3)
}


def execute_for_rate(dataset: Dataset,
                     val_probs: np.ndarray,
                     val_labels: np.ndarray,
                     test_probs: np.ndarray,
                     data_iterator_name: str,
                     window_size: Optional[int],
                     noise_rate: Optional[float],
                     rates: List[float],
                     model_path: str,
                     strategy: ExitStrategy,
                     num_reps: int,
                     num_test_reps: int,
                     max_num_samples: Optional[int]) -> Dict[str, Dict[str, Any]]:
    # Make the exit policy
    policy = make_policy(strategy=strategy, rates=rates, model_path=model_path)
    policy.fit(val_probs=val_probs, val_labels=val_labels)

    # Run the policy on the validation and test sets
    val_iterator = make_data_iterator(name=data_iterator_name,
                                      dataset=dataset,
                                      pred_probs=val_probs,
                                      window_size=window_size,
                                      noise_rate=noise_rate,
                                      num_reps=num_reps,
                                      fold='val')
    val_result = policy.test(data_iterator=val_iterator,
                             max_num_samples=max_num_samples)

    test_iterator = make_data_iterator(name=data_iterator_name,
                                       dataset=dataset,
                                       pred_probs=test_probs,
                                       window_size=window_size,
                                       noise_rate=noise_rate,
                                       num_reps=num_test_reps,
                                       fold='test')
    test_result = policy.test(data_iterator=test_iterator,
                              max_num_samples=max_num_samples)

    result = dict(val=dict(), test=dict())
    summary = dict(val=dict(), test=dict())

    val_dict = {
        'preds': val_result.predictions.tolist(),
        'output_levels': val_result.output_levels.tolist(),
        'labels': val_result.labels.tolist(),
        'monitor_stats': val_result.monitor_stats,
        'window_size': window_size,
        'num_reps': num_reps
    }

    test_dict = {
        'preds': test_result.predictions.tolist(),
        'output_levels': test_result.output_levels.tolist(),
        'labels': test_result.labels.tolist(),
        'monitor_stats': test_result.monitor_stats,
        'window_size': window_size,
        'num_reps': num_reps
    }

    #prob_bias = test_result.monitor_stats['prob_bias']
    #first_biases = [bias[0] for bias in prob_bias]
    #print('\nAverage Prob Bias: {:.4f}'.format(np.average(first_biases)))

    result['val'][val_iterator.name] = val_dict
    result['test'][test_iterator.name] = test_dict

    val_summary: Dict[str, float] = dict()
    test_summary: Dict[str, float] = dict()

    for metric in InferenceMetric:
        if metric in (InferenceMetric.NGRAM_MI, InferenceMetric.COUNT_NGRAM_MI):
            continue

        val_summary[metric.name.lower()] = compute_metric(preds=val_result.predictions,
                                                          exit_decisions=val_result.output_levels,
                                                          labels=val_result.labels,
                                                          metric=metric,
                                                          window_size=min(window_size, 10),
                                                          num_outputs=len(rates))
        test_summary[metric.name.lower()] = compute_metric(preds=test_result.predictions,
                                                           exit_decisions=test_result.output_levels,
                                                           labels=test_result.labels,
                                                           metric=metric,
                                                           window_size=min(window_size, 10),
                                                           num_outputs=len(rates))

    summary['val'][val_iterator.name] = val_summary
    summary['test'][test_iterator.name] = test_summary

    return result, summary


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--dataset-order', type=str, required=True, help='The type of data iterator to create.')
    parser.add_argument('--reps', type=int, default=1, help='The number of repetitions of the dataset.')
    parser.add_argument('--trials', type=int, default=1, help='The number of independent trials.')
    parser.add_argument('--window-size', type=int, help='The window size used to build the dataset.')
    parser.add_argument('--noise-rate', type=float, help='The optional noise rate to use for the dataset order.')
    parser.add_argument('--max-num-samples', type=int, help='Optional maximum number of samples (for testing)')
    parser.add_argument('--should-approx-softmax', action='store_true', help='Whether to use an approximate softmax function to mimic fixed point arithmetic.')
    parser.add_argument('--num-test-reps', type=int, default=1, help='The number of repetitions to apply to the test set. In almost all cases, this should be 1.')
    args = parser.parse_args()

    assert args.reps >= 1, 'Must provide a positive number of dataset repititions'
    assert args.trials >= 1, 'Must provide a positive number of trials'

    # Restore the model
    model: BaseClassifier = restore_classifier(model_path=args.model_path, model_mode=ModelMode.TEST)

    # Get the predictions from the models
    val_probs = model.validate(should_approx=args.should_approx_softmax)  # [B, L, K]
    test_probs = model.test(should_approx=args.should_approx_softmax)  # [C, L, K]

    # Get the validation labels (we use this to fit any policies)
    val_labels = model.dataset.get_val_labels()  # [B]

    # Make the target rates
    (lower_bound, upper_bound) = TARGET_BOUNDS[model.num_outputs]
    single_rates = list(np.arange(lower_bound, upper_bound + 0.01, 0.05))
    rand = np.random.RandomState(seed=591)

    rates = get_exit_rates(single_rates=single_rates, num_outputs=model.num_outputs)

    # Execute all early stopping policies
    strategies = list(ExitStrategy)

    # Load the existing test log (if present)
    file_name = os.path.basename(args.model_path).split('.')[0]
    output_folder_path = os.path.join(os.path.dirname(args.model_path), '{}_test-logs'.format(file_name))
    make_dir(output_folder_path)
    
    for strategy in strategies:
        num_trials = args.trials if POLICY_HAS_RANDOMNESS[strategy] else 1

        for trial in range(num_trials):
            strategy_name = strategy.name.lower()

            # Read in the old log (if it exists)
            test_log_name = '{}-trial{}.json.gz'.format(strategy_name, trial)
            test_log_path = os.path.join(output_folder_path, test_log_name)

            summary_log_name = '{}-trial{}-summary.json.gz'.format(strategy_name, trial)
            summary_log_path = os.path.join(output_folder_path, summary_log_name)

            results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
                'val': dict(),
                'test': dict()
            }

            result_summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
                'val': dict(),
                'test': dict()
            }

            if os.path.exists(test_log_path):
                results = read_json_gz(test_log_path)

            if os.path.exists(summary_log_path):
                result_summary = read_json_gz(summary_log_path)

            for rate_list in rates:
                rate_key = ' '.join('{:.2f}'.format(round(r, 2)) for r in rate_list)
                print('Testing {} on {} for trial {}'.format(strategy_name.capitalize(), rate_key, trial + 1), end='\r')

                rate_result, rate_summary = execute_for_rate(dataset=model.dataset,
                                                             val_probs=val_probs,
                                                             val_labels=val_labels,
                                                             test_probs=test_probs,
                                                             rates=rate_list,
                                                             model_path=args.model_path,
                                                             strategy=strategy,
                                                             data_iterator_name=args.dataset_order,
                                                             window_size=args.window_size,
                                                             noise_rate=args.noise_rate,
                                                             num_reps=args.reps,
                                                             num_test_reps=args.num_test_reps,
                                                             max_num_samples=args.max_num_samples)

                # Log the results
                for fold in ['val', 'test']:
                    if rate_key not in results[fold]:
                        results[fold][rate_key] = dict()

                    for dataset_order in rate_result[fold].keys():
                        results[fold][rate_key][dataset_order] = rate_result[fold][dataset_order]

                    if rate_key not in result_summary[fold]:
                        result_summary[fold][rate_key] = dict()

                    for dataset_order in rate_summary[fold].keys():
                        result_summary[fold][rate_key][dataset_order] = rate_summary[fold][dataset_order]

            print()

            # Save the results into the test logs
            save_json_gz(results, test_log_path)
            save_json_gz(result_summary, summary_log_path)
