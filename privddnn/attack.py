import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from typing import DefaultDict, List, Tuple, Dict

from privddnn.attack.attack_dataset import make_similar_dataset, make_noisy_dataset, make_sequential_dataset
from privddnn.attack.attack_classifiers import DecisionTreeEnsembleCount, DecisionTreeEnsembleNgram, MostFrequentClassifier, LogisticRegressionCount
from privddnn.classifier import BaseClassifier, ModelMode, OpName
from privddnn.exiting import ALL_POLICY_NAMES
from privddnn.restore import restore_classifier
from privddnn.utils.file_utils import read_json_gz, save_json_gz, save_pickle_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train-model-path', type=str, required=True, help='Path to the neural network (h5 file) which will be used to train this attack model.')
    parser.add_argument('--eval-log-folder', type=str, help='Path to the test logs which we will evaluate the attack model on. If not included, defauls to that of the training neural network.')
    parser.add_argument('--dataset-order', type=str, required=True, help='The dataset order used to train / evaluate the attack model. Should include the window size (e.g., same-label-10)')
    parser.add_argument('--policy-names', type=str, required=True, nargs='+', help='The names of the policies to evaluate. Can be `all`.')
    parser.add_argument('--train-policy', type=str, choices=['max_prob', 'entropy'], help='The policy used to train the attack model. If none, will default to using the same policies for training and evaluation.')
    parser.add_argument('--trials', type=int, default=1, required=True, help='The number of trials to fit attack models for.')
    parser.add_argument('--window-size', type=int, help='Optional override of the window size. If not included, defaults to the window size of the dataset order.')
    parser.add_argument('--should-print', action='store_true', help='Whether to print debugging information.')
    args = parser.parse_args()

    # Get the path to the training log (default to eval log if needed)
    train_log_folder_path = args.train_model_path.replace('.h5', '_test-logs')
    eval_log_folder_path = train_log_folder_path if args.eval_log_folder is None else args.eval_log_folder

    # Maps policy name -> { clf type -> [accuracy] }
    train_attack_results: Dict[str, DefaultDict[str, List[float]]] = dict()
    test_attack_results: Dict[str, DefaultDict[str, List[float]]] = dict()

    # Fit the most-frequent classifier based on the original model
    model: BaseClassifier = restore_classifier(model_path=args.train_model_path, model_mode=ModelMode.TEST)
    num_labels = model.dataset.num_labels

    train_policy_name = args.train_policy if args.train_policy is not None else 'same'
    train_log_path_tokens = train_log_folder_path.split(os.sep)
    train_log_name = train_log_path_tokens[-1] if len(train_log_path_tokens[-1]) > 0 else train_log_path_tokens[-2]
    attack_key = '{}_{}'.format(train_log_name.replace('_test-logs', ''), train_policy_name)

    policy_names = args.policy_names if ('all' not in args.policy_names) else ALL_POLICY_NAMES

    for trial in range(args.trials):
        for policy_name in policy_names:
            # Initialize the result dictionaries for this policy
            train_attack_results: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
            test_attack_results: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

            # Get the exit decisions and predictions based on the test logs
            train_policy_name = policy_name if args.train_policy is None else args.train_policy
            train_log_path = os.path.join(train_log_folder_path, '{}-trial{}.json.gz'.format(train_policy_name, trial))
            eval_log_path = os.path.join(eval_log_folder_path, '{}-trial{}.json.gz'.format(policy_name, trial))

            if not os.path.exists(train_log_path):
                if args.should_print:
                    print('No file named {}. Skipping...'.format(train_log_path))
                continue

            if not os.path.exists(eval_log_path):
                if args.should_print:
                    print('No file named {}. Skipping...'.format(eval_log_path))
                continue

            train_log = read_json_gz(train_log_path)['val']
            eval_log = read_json_gz(eval_log_path)['test']

            # Read in the existing output path (if possible) to append to any existing results
            output_path = os.path.join(eval_log_folder_path, '{}-attack-trial{}.json.gz'.format(policy_name, trial))
            if os.path.exists(output_path):
                output_log = read_json_gz(output_path)
            else:
                output_log = dict()

            for rate in train_log.keys():
                window_size = train_log[rate][args.dataset_order]['window_size'] if args.window_size is None else args.window_size

                # Get the results from training and validation
                train_levels = train_log[rate][args.dataset_order]['output_levels']
                train_preds = train_log[rate][args.dataset_order]['preds']

                test_levels = eval_log[rate][args.dataset_order]['output_levels']
                test_preds = eval_log[rate][args.dataset_order]['preds']

                # Build the attack datasets
                train_attack_inputs, train_attack_outputs = make_sequential_dataset(levels=train_levels,
                                                                                    preds=train_preds,
                                                                                    window_size=window_size)

                test_attack_inputs, test_attack_outputs = make_sequential_dataset(levels=test_levels,
                                                                                  preds=test_preds,
                                                                                  window_size=window_size)

                rate_str = ' '.join('{:.2f}'.format(round(float(r), 2)) for r in rate.split(' '))

                if args.should_print:
                    print('Starting {} on {}. # Train: {}, # Test: {}'.format(policy_name, rate_str, len(train_attack_inputs), len(test_attack_inputs)), end='\r')

                # Fit and evaluate the classifiers
                if model.num_outputs == 2:
                    classifiers = [DecisionTreeEnsembleCount(), DecisionTreeEnsembleNgram(), MostFrequentClassifier(), LogisticRegressionCount()]
                else:
                    classifiers = [DecisionTreeEnsembleCount(), DecisionTreeEnsembleNgram()]

                for clf in classifiers:
                    clf.fit(train_attack_inputs, train_attack_outputs, num_labels=num_labels)
                
                    train_acc = clf.score(train_attack_inputs, train_attack_outputs)
                    test_acc = clf.score(test_attack_inputs, test_attack_outputs)

                    train_attack_results[clf.name][rate_str] = train_acc
                    test_attack_results[clf.name][rate_str] = test_acc

                    #if rate.startswith('0.5') and trial == 0 and (isinstance(clf, DecisionTreeEnsembleCount)):
                    #    model_output_path = 'saved_models/uci_har/30-04-2023/{}-attack-model.pkl.gz'.format(policy_name)
                    #    clf.save(model_output_path)

            if args.should_print:
                print()

            if attack_key not in output_log:
                output_log[attack_key] = dict()

            if args.dataset_order not in output_log[attack_key]:
                output_log[attack_key][args.dataset_order] = dict()

            # Save the results
            output_log[attack_key][args.dataset_order]['attack_test'] = test_attack_results
            output_log[attack_key][args.dataset_order]['attack_train'] = train_attack_results
            output_log[attack_key][args.dataset_order]['attack_train_log'] = train_log_path
            output_log[attack_key][args.dataset_order]['attack_policy'] = train_policy_name

            save_json_gz(output_log, output_path)
