import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from typing import DefaultDict, List, Tuple, Dict

from privddnn.attack.attack_dataset import make_similar_dataset, make_noisy_dataset, make_sequential_dataset
from privddnn.attack.attack_classifiers import MostFrequentClassifier, MajorityClassifier, LogisticRegressionCount, LogisticRegressionNgram, NgramClassifier
from privddnn.attack.attack_classifiers import WindowNgramClassifier
from privddnn.classifier import BaseClassifier, ModelMode, OpName
from privddnn.restore import restore_classifier
from privddnn.utils.file_utils import read_json_gz, save_json_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train-model-path', type=str, required=True)
    parser.add_argument('--eval-log-folder', type=str)
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--policy-names', type=str, required=True, nargs='+')
    parser.add_argument('--train-policy', type=str, choices=['max_prob', 'entropy'])
    parser.add_argument('--trials', type=int, default=1, required=True)
    parser.add_argument('--should-print', action='store_true')
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

    for trial in range(args.trials):
        for policy_name in args.policy_names:
            # Initialize the result dictionaries for this policy
            train_attack_results: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
            test_attack_results: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

            # Get the exit decisions and predictions based on the test logs. For now, we always use trial 0.
            train_policy_name = policy_name if args.train_policy is None else args.train_policy
            train_log_path = os.path.join(train_log_folder_path, '{}-trial{}.json.gz'.format(train_policy_name, trial))
            train_log = read_json_gz(train_log_path)['val']

            eval_log_path = os.path.join(eval_log_folder_path, '{}-trial{}.json.gz'.format(policy_name, trial))
            eval_log = read_json_gz(eval_log_path)['test']

            output_path = os.path.join(eval_log_folder_path, '{}-attack-trial{}.json.gz'.format(policy_name, trial))
            if os.path.exists(output_path):
                output_log = read_json_gz(output_path)
            else:
                output_log = dict()

            for rate in train_log.keys():
                window_size = train_log[rate][args.dataset_order]['window_size']

                # Get the results from training and validation
                val_levels = train_log[rate][args.dataset_order]['output_levels']
                val_preds = train_log[rate][args.dataset_order]['preds']

                test_levels = eval_log[rate][args.dataset_order]['output_levels']
                test_preds = eval_log[rate][args.dataset_order]['preds']

                # Build the attack datasets
                train_attack_inputs, train_attack_outputs = make_sequential_dataset(levels=val_levels,
                                                                                    preds=val_preds,
                                                                                    window_size=window_size)

                test_attack_inputs, test_attack_outputs = make_sequential_dataset(levels=test_levels,
                                                                                  preds=test_preds,
                                                                                  window_size=window_size)

                rate_str = ' '.join('{:.2f}'.format(round(float(r), 2)) for r in rate.split(' '))

                if args.should_print:
                    print('Starting {} on {}. # Train: {}, # Test: {}'.format(policy_name, rate_str, len(train_attack_inputs), len(test_attack_inputs)), end='\n')

                # Fit and evaluate the majority attack classifier
                majority_clf = MajorityClassifier()
                majority_clf.fit(train_attack_inputs, train_attack_outputs, num_labels=num_labels)

                train_acc = majority_clf.score(train_attack_inputs, train_attack_outputs)
                test_acc = majority_clf.score(test_attack_inputs, test_attack_outputs)

                #print('Majority: Train Accuracy: {:.5f}, Test Accuracy: {:.5f}'.format(train_acc['accuracy'], test_acc['accuracy']))
                #print('Majority Avg Correct Rank: Train Accuracy: {:.5f}, Test Accuracy: {:.5f}'.format(train_acc['correct_rank'], test_acc['correct_rank']))

                train_attack_results[majority_clf.name][rate_str] = train_acc
                test_attack_results[majority_clf.name][rate_str] = test_acc

                wngram_clf = WindowNgramClassifier(window_size=5, num_neighbors=5)
                wngram_clf.fit(train_attack_inputs, train_attack_outputs, num_labels=num_labels)

                train_acc = wngram_clf.score(train_attack_inputs, train_attack_outputs)
                test_acc = wngram_clf.score(test_attack_inputs, test_attack_outputs)

                #print('WNgram: Train Accuracy: {:.5f}, Test Accuracy: {:.5f}'.format(train_acc['accuracy'], test_acc['accuracy']))

                train_attack_results[wngram_clf.name][rate_str] = train_acc
                test_attack_results[wngram_clf.name][rate_str] = test_acc

                # Fit and evaluate the NGram classifier
                ngram_clf = NgramClassifier(num_neighbors=1)
                ngram_clf.fit(train_attack_inputs, train_attack_outputs, num_labels=num_labels)

                train_acc = ngram_clf.score(train_attack_inputs, train_attack_outputs)
                test_acc = ngram_clf.score(test_attack_inputs, test_attack_outputs)

                #print('NGram: Train Accuracy: {:.5f}, Test Accuracy: {:.5f}'.format(train_acc['accuracy'], test_acc['accuracy']))

                train_attack_results[ngram_clf.name][rate_str] = train_acc
                test_attack_results[ngram_clf.name][rate_str] = test_acc

                # Fit and evaluate the logistic regression classifiers
                lr_clf = LogisticRegressionCount()
                lr_clf.fit(train_attack_inputs, train_attack_outputs, num_labels=num_labels)

                print('\nLogistic Regression Count')
                train_acc = lr_clf.score(train_attack_inputs, train_attack_outputs)
                test_acc = lr_clf.score(test_attack_inputs, test_attack_outputs)
                print('==========')

                train_attack_results[lr_clf.name][rate_str] = train_acc
                test_attack_results[lr_clf.name][rate_str] = test_acc

                lr_path = 'jetson_attack_models/{}_{}.pkl.gz'.format(policy_name, rate_str.replace('.', '-').replace(' ', '_'))
                lr_clf.save(lr_path)

                lrn_clf = LogisticRegressionNgram()
                lrn_clf.fit(train_attack_inputs, train_attack_outputs, num_labels=num_labels)

                train_acc = lrn_clf.score(train_attack_inputs, train_attack_outputs)
                test_acc = lrn_clf.score(test_attack_inputs, test_attack_outputs)

                train_attack_results[lrn_clf.name][rate_str] = train_acc
                test_attack_results[lrn_clf.name][rate_str] = test_acc

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
