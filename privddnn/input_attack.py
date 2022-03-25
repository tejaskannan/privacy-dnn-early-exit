import os
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from typing import DefaultDict, List, Tuple, Dict

from privddnn.attack.attack_dataset import make_input_sequential_dataset
from privddnn.attack.input_generators import MajorityGenerator
from privddnn.classifier import BaseClassifier, ModelMode
from privddnn.restore import restore_classifier
from privddnn.utils.file_utils import read_json_gz, save_json_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train-model-path', type=str, required=True)
    parser.add_argument('--eval-log', type=str)
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--train-policy', type=str, choices=['max_prob', 'entropy'])
    args = parser.parse_args()

    # Get the path to the training log (default to eval log if needed)
    train_log_path = args.train_model_path.replace('.h5', '_test-log.json.gz')
    eval_log_path = train_log_path if args.eval_log is None else args.eval_log

    # Read the logs for training and testing
    train_log = read_json_gz(train_log_path)
    eval_log = read_json_gz(eval_log_path)

    policy_names = ['random', 'max_prob', 'adaptive_random_max_prob']

    # Maps policy name -> { clf type -> [accuracy] }
    train_attack_results: Dict[str, DefaultDict[str, List[Dict[str, float]]]] = dict()
    test_attack_results: Dict[str, DefaultDict[str, List[Dict[str, float]]]] = dict()

    # Fit the most-frequent classifier based on the original model
    model: BaseClassifier = restore_classifier(model_path=args.train_model_path, model_mode=ModelMode.TEST)

    key = next(iter(train_log['val']['random'].keys()))
    window_size = train_log['val']['random'][key][args.dataset_order]['window_size']
    order_name = '-'.join(args.dataset_order.split('-')[0:-1])

    for policy_name in policy_names:
        # Initialize the top-level dictionaries
        train_attack_results[policy_name] = defaultdict(list)
        test_attack_results[policy_name] = defaultdict(list)

        if policy_name != 'max_prob':
            continue

        for rate in sorted(train_log['val'][policy_name].keys()):
            train_policy_name = policy_name if args.train_policy is None else args.train_policy

            train_decisions = train_log['val'][train_policy_name][rate][args.dataset_order]['output_levels']
            test_decisions = eval_log['test'][policy_name][rate][args.dataset_order]['output_levels']

            # Build the attack datasets
            train_attack_decisions, train_attack_data, train_attack_labels = make_input_sequential_dataset(dataset=model.dataset,
                                                                                                           dataset_order=order_name,
                                                                                                           exit_decisions=train_decisions,
                                                                                                           fold='val',
                                                                                                           num_exits=model.num_outputs,
                                                                                                           window_size=window_size)

            test_attack_decisions, test_attack_data, _ = make_input_sequential_dataset(dataset=model.dataset,
                                                                                       dataset_order=order_name,
                                                                                       exit_decisions=test_decisions,
                                                                                       fold='test',
                                                                                       num_exits=model.num_outputs,
                                                                                       window_size=window_size)

            rate_str = ' '.join('{:.2f}'.format(round(float(r), 2)) for r in rate.split(' '))
            print('Starting {} on {}. # Train: {}, # Test: {}'.format(policy_name, rate_str, train_attack_data.shape, test_attack_data.shape), end='\n')

            # Fit and evaluate the majority attack classifier
            majority_model = MajorityGenerator()
            majority_model.fit(train_attack_decisions, train_attack_data, train_attack_labels)

            train_error = majority_model.score(train_attack_decisions, train_attack_data)
            test_error = majority_model.score(test_attack_decisions, test_attack_data)

            train_error['exit_rates'] = rate
            test_error['exit_rates'] = rate

            print('Majority: Train Error: {:.5f}, Test Error: {:.5f}'.format(train_error['l1_error'], test_error['l1_error']))
            print('Majority Weighted: Train Error: {:.5f}, Test Error: {:.5f}'.format(train_error['weighted_l1_error'], test_error['weighted_l1_error']))

            train_attack_results[policy_name][majority_model.name].append(train_error)
            test_attack_results[policy_name][majority_model.name].append(test_error)

        print()

    # The attack log uses a dictionary of dictionaries. The top-level key is the
    # train model type, and the value is the attack results for this configuration
    train_policy_name = args.train_policy if args.train_policy is not None else 'same'
    train_log_name = os.path.basename(train_log_path)
    attack_key = '{}_{}'.format(train_log_name.replace('_test-log.json.gz', ''), train_policy_name)

    if attack_key not in eval_log:
        eval_log[attack_key] = dict()

    if args.dataset_order not in eval_log[attack_key]:
        eval_log[attack_key][args.dataset_order] = dict()

    # Save the results
    eval_log[attack_key][args.dataset_order]['input_attack_test'] = test_attack_results
    eval_log[attack_key][args.dataset_order]['input_attack_train'] = train_attack_results
    eval_log[attack_key][args.dataset_order]['input_attack_train_log'] = train_log_path
    save_json_gz(eval_log, eval_log_path)
