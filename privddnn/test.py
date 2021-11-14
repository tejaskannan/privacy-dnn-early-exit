import numpy as np
import os.path
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum, auto
from typing import List, Tuple, DefaultDict, Dict, Optional

from privddnn.dataset import Dataset
from privddnn.dataset.data_iterators import NearestNeighborIterator
from privddnn.classifier import BaseClassifier, ModelMode, OpName
from privddnn.exiting.early_exit import ExitStrategy, EarlyExiter, make_policy, EarlyExitResult
from privddnn.restore import restore_classifier
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info, compute_target_exit_rates, compute_stop_counts
from privddnn.utils.plotting import to_label
from privddnn.utils.file_utils import save_json_gz


def execute_for_rate(dataset: Dataset,
                     clf: BaseClassifier,
                     val_probs: np.ndarray,
                     val_labels: np.ndarray,
                     window_size: int,
                     pred_rates: np.ndarray,
                     rate: float,
                     model_path: str,
                     strategy: ExitStrategy,
                     num_trials: int,
                     max_num_samples: Optional[int]) -> Dict[str, Dict[str, List[float]]]:
    # Make the exit policy
    rates = [rate, 1.0 - rate]
    policy = make_policy(strategy=strategy, rates=rates, model_path=model_path)
    policy.fit(val_probs=val_probs, val_labels=val_labels)

    # Run the policy on the validation and test sets
    val_iterator = NearestNeighborIterator(dataset=dataset, clf=clf, window_size=window_size, num_trials=num_trials, fold='val')
    val_result = policy.test(data_iterator=val_iterator,
                             num_labels=dataset.num_labels,
                             pred_rates=pred_rates,
                             max_num_samples=max_num_samples)

    test_iterator = NearestNeighborIterator(dataset=dataset, clf=clf, window_size=window_size, num_trials=num_trials, fold='test')
    test_result = policy.test(data_iterator=test_iterator,
                              num_labels=dataset.num_labels,
                              pred_rates=pred_rates,
                              max_num_samples=max_num_samples)

    result = dict(val=list(), test=list())

    val_dict = {
        'preds': val_result.predictions.tolist(),
        'output_levels': val_result.output_levels.tolist(),
        'labels': val_result.labels.tolist(),
        'num_changed': val_result.num_changed,
        'selection_counts': { key.name: value for key, value in val_result.selection_counts.items() },
        'window_size': window_size,
        'num_trials': num_trials
    }

    test_dict = {
        'preds': test_result.predictions.tolist(),
        'output_levels': test_result.output_levels.tolist(),
        'labels': test_result.labels.tolist(),
        'num_changed': test_result.num_changed,
        'selection_counts': { key.name: value for key, value in test_result.selection_counts.items() },
        'window_size': window_size,
        'num_trials': num_trials
    }

    result['val'].append(val_dict)
    result['test'].append(test_dict)

    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--trials', type=int, required=True, help='The number of trials per policy.')
    parser.add_argument('--window-size', type=int, required=True, help='The window size used to build the dataset.')
    parser.add_argument('--max-num-samples', type=int, help='Optional maximum number of samples (for testing)')
    args = parser.parse_args()

    assert args.trials >= 1, 'Must provide a positive number of trials'
    assert args.window_size >= 1, 'The window size must be positive'

    # Restore the model
    model: BaseClassifier = restore_classifier(model_path=args.model_path, model_mode=ModelMode.TEST)

    # Get the predictions from the models
    test_probs = model.test(op=OpName.PROBS)  # [C, L, K]
    val_probs = model.validate(op=OpName.PROBS)  # [B, L, K]

    # Get the validation labels (we use this to fit any policies)
    val_labels = model.dataset.get_val_labels()  # [B]
    stop_counts = compute_stop_counts(probs=val_probs)

    rates = list(sorted(np.arange(0.0, 1.01, 0.05)))
    rand = np.random.RandomState(seed=591)

    #rates = [0.5, 0.6]

    # Execute all early stopping policies
    results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = dict(val=dict(), test=dict())

    #strategies = [ExitStrategy.MAX_PROB, ExitStrategy.ENTROPY, ExitStrategy.LABEL_MAX_PROB, ExitStrategy.LABEL_ENTROPY, ExitStrategy.HYBRID_MAX_PROB, ExitStrategy.HYBRID_ENTROPY, ExitStrategy.RANDOM]
    #strategies = [ExitStrategy.RANDOM, ExitStrategy.GREEDY_EVEN, ExitStrategy.MAX_PROB, ExitStrategy.LABEL_MAX_PROB, ExitStrategy.EVEN_MAX_PROB]
    strategies = [ExitStrategy.RANDOM, ExitStrategy.EVEN_MAX_PROB, ExitStrategy.GREEDY_EVEN, ExitStrategy.MAX_PROB]

    for strategy in strategies:
        strategy_name = strategy.name.lower()
        val_results: Dict[str, Dict[str, List[float]]] = dict()
        test_results: Dict[str, Dict[str, List[float]]] = dict()

        print('Testing {}'.format(strategy_name.capitalize()))

        for rate in reversed(rates):
            rate_result = execute_for_rate(dataset=model.dataset,
                                           clf=model,
                                           val_probs=val_probs,
                                           val_labels=val_labels,
                                           pred_rates=stop_counts,
                                           rate=rate,
                                           model_path=args.model_path,
                                           strategy=strategy,
                                           window_size=args.window_size,
                                           num_trials=args.trials,
                                           max_num_samples=args.max_num_samples)

            # Log the results
            rate_key = str(round(rate, 2))
            val_results[rate_key] = rate_result['val']
            test_results[rate_key] = rate_result['test']

        results['val'][strategy_name] = val_results
        results['test'][strategy_name] = test_results

    # Save the results into the test log
    file_name = os.path.basename(args.model_path).split('.')[0]
    test_log_name = '{}_test-log.json.gz'.format(file_name)
    test_log_path = os.path.join(os.path.dirname(args.model_path), test_log_name)

    save_json_gz(results, test_log_path)
