import numpy as np
import os.path
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum, auto
from itertools import permutations
from typing import List, Tuple, DefaultDict, Dict, Optional

from privddnn.dataset import Dataset
from privddnn.dataset.data_iterators import make_data_iterator
from privddnn.classifier import BaseClassifier, ModelMode, OpName
from privddnn.exiting import ExitStrategy, EarlyExiter, make_policy, EarlyExitResult
from privddnn.restore import restore_classifier
from privddnn.utils.file_utils import save_json_gz, read_json_gz
from privddnn.test import execute_for_rate


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

    single_rates = list(sorted(np.arange(0.3, 0.71, 0.05)))
    rand = np.random.RandomState(seed=591)

    # Execute all early stopping policies
    strategies = [ExitStrategy.ADAPTIVE_RANDOM_MAX_PROB, ExitStrategy.MAX_PROB, ExitStrategy.RANDOM]

    # Load the existing test log (if present)
    file_name = os.path.basename(args.model_path).split('.')[0]
    test_log_name = '{}_test-log.json.gz'.format(file_name)
    test_log_path = os.path.join(os.path.dirname(args.model_path), test_log_name)

    results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = dict(val=dict(), test=dict())
    if os.path.exists(test_log_path):
        results = read_json_gz(test_log_path)

    # Get the (unique) rate keys
    rate_settings: List[Tuple[float, ...]] = list(permutations(single_rates, model.num_outputs - 1))

    rates: List[float] = []
    for setting in rate_settings:
        last_rate = 1.0 - sum(setting)
        if last_rate >= 0.0:
            rates.append(list(setting) + [last_rate])

    for strategy in strategies:
        strategy_name = strategy.name.lower()

        if strategy_name not in results['val']:
            results['val'][strategy_name] = dict()

        if strategy_name not in results['test']:
            results['test'][strategy_name] = dict()

        for rate_list in rates:
            rate_key = ' '.join('{:.2f}'.format(round(r, 2)) for r in rate_list)
            print('Testing {} on {}'.format(strategy_name.capitalize(), rate_key), end='\r')

            rate_result = execute_for_rate(dataset=model.dataset,
                                           val_probs=val_probs,
                                           val_labels=val_labels,
                                           test_probs=test_probs,
                                           rates=rate_list,
                                           model_path=args.model_path,
                                           strategy=strategy,
                                           data_iterator_name=args.dataset_order,
                                           window_size=args.window_size,
                                           num_reps=args.reps,
                                           max_num_samples=args.max_num_samples)

            # Log the results
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
