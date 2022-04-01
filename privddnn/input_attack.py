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
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--policy-names', type=str, required=True, nargs='+')
    parser.add_argument('--train-policy', type=str, choices=['max_prob', 'entropy'])
    parser.add_argument('--trials', type=int, default=1, required=True)
    args = parser.parse_args()

    # Get the path to the training log (default to eval log if needed)
    log_folder = args.model_path.replace('.h5', '_test-logs')

    # Maps policy name -> { clf type -> [accuracy] }
    train_attack_results: Dict[str, DefaultDict[str, List[Dict[str, float]]]] = dict()
    test_attack_results: Dict[str, DefaultDict[str, List[Dict[str, float]]]] = dict()

    # Fit the most-frequent classifier based on the original model
    model: BaseClassifier = restore_classifier(model_path=args.model_path, model_mode=ModelMode.TEST)

    order_name = '-'.join(args.dataset_order.split('-')[0:-1])

    for trial in range(args.trials):
        for policy_name in args.policy_names:
            # Initialize the top-level dictionaries
            train_attack_results: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
            test_attack_results: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

            train_policy_name = policy_name if args.train_policy is None else args.train_policy
            train_log_path = os.path.join(log_folder, '{}-trial{}.json.gz'.format(train_policy_name, trial))
            train_log = read_json_gz(train_log_path)['val']

            eval_log_path = os.path.join(log_folder, '{}-trial{}.json.gz'.format(policy_name, trial))
            eval_log = read_json_gz(eval_log_path)['test']

            output_path = os.path.join(log_folder, '{}-attack_input-trial{}.json.gz'.format(policy_name, trial))
            if os.path.exists(output_path):
                output_log = read_json_gz(output_path)
            else:
                output_log = dict()

            for rate in sorted(train_log.keys()):
                window_size = int(train_log[rate][args.dataset_order]['window_size'])
                train_decisions = train_log[rate][args.dataset_order]['output_levels']
                test_decisions = eval_log[rate][args.dataset_order]['output_levels']

                # Build the attack datasets
                train_attack_decisions, train_attack_data, train_attack_labels = make_input_sequential_dataset(dataset=model.dataset,
                                                                                                               dataset_order=order_name,
                                                                                                               exit_decisions=train_decisions,
                                                                                                               fold='val',
                                                                                                               num_exits=model.num_outputs,
                                                                                                               window_size=window_size)

                test_attack_decisions, test_attack_data, test_attack_labels = make_input_sequential_dataset(dataset=model.dataset,
                                                                                                            dataset_order=order_name,
                                                                                                            exit_decisions=test_decisions,
                                                                                                            fold='test',
                                                                                                            num_exits=model.num_outputs,
                                                                                                            window_size=window_size)

                rate_str = ' '.join('{:.2f}'.format(round(float(r), 2)) for r in rate.split(' '))
                print('Starting {} on {}. # Train: {}, # Test: {}'.format(policy_name, rate_str, train_attack_data.shape[0], test_attack_data.shape[0]), end='\r')

                # Fit and evaluate the majority attack classifier
                majority_model = MajorityGenerator()
                majority_model.fit(train_attack_decisions, train_attack_data, train_attack_labels)

                train_error = majority_model.score(train_attack_decisions, train_attack_data, train_attack_labels)
                test_error = majority_model.score(test_attack_decisions, test_attack_data, test_attack_labels)

                train_error['exit_rates'] = rate
                test_error['exit_rates'] = rate

                #print('Majority: Train Error: {:.5f}, Test Error: {:.5f}'.format(train_error['l1_error'], test_error['l1_error']))
                #print('Majority Weighted: Train Error: {:.5f}, Test Error: {:.5f}'.format(train_error['weighted_l1_error'], test_error['weighted_l1_error']))

                train_attack_results[majority_model.name][rate] = train_error
                test_attack_results[majority_model.name][rate] = test_error

            print()

            # Save the result in the output log
            if args.dataset_order not in output_log:
                output_log[args.dataset_order] = dict()

            output_log[args.dataset_order]['train'] = train_attack_results
            output_log[args.dataset_order]['test'] = test_attack_results
            save_json_gz(output_log, output_path)
