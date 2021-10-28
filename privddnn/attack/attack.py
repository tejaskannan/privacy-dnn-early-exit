import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from typing import DefaultDict, List, Tuple, Dict

from privddnn.attack.attack_dataset import make_similar_dataset, make_noisy_dataset
from privddnn.attack.attack_classifiers import MostFrequentClassifier, MajorityClassifier, LogisticRegressionClassifier
from privddnn.classifier import BaseClassifier, ModelMode, OpName
from privddnn.restore import restore_classifier
from privddnn.utils.file_utils import read_json_gz, save_json_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train-model-path', type=str, required=True)
    parser.add_argument('--eval-log', type=str)
    parser.add_argument('--train-policy', type=str, choices=['max_prob', 'entropy'])
    args = parser.parse_args()

    # Get the path to the training log (default to eval log if needed)
    train_log_path = args.train_model_path.replace('.pkl.gz', '_test-log.json.gz')
    eval_log_path = train_log_path if args.eval_log is None else args.eval_log

    # Read the logs for training and testing
    train_log = read_json_gz(train_log_path)
    eval_log = read_json_gz(eval_log_path)

    # Get paths to the 'similarity' indices
    dataset_name = args.train_model_path.split(os.sep)[-3]
    val_index_path = '../data/{}/val'.format(dataset_name)
    test_index_path = '../data/{}/test'.format(dataset_name)

    rates = [str(round(r / 20.0, 2)) for r in range(21)]
    #policy_names = ['random', 'max_prob', 'label_max_prob', 'hybrid_max_prob', 'entropy', 'label_entropy', 'hybrid_entropy']
    policy_names = ['random', 'even_max_prob', 'max_prob']

    window_size = 25
    noise_rate = 0.2
    num_trials = 1

    # Maps policy name -> { clf type -> [accuracy] }
    train_attack_results: Dict[str, DefaultDict[str, List[float]]] = dict()
    test_attack_results: Dict[str, DefaultDict[str, List[float]]] = dict()

    # Fit the most-frequent classifier based on the original model
    model: BaseClassifier = restore_classifier(model_path=args.train_model_path, model_mode=ModelMode.TEST)
    val_probs = model.validate(op=OpName.PROBS)  # [B, L, K]
    val_preds = np.argmax(val_probs, axis=-1)  # [B, L]

    most_freq_clf = MostFrequentClassifier(window=window_size, num_labels=np.max(val_preds) + 1)
    most_freq_clf.fit(inputs=val_preds, labels=val_preds)

    for policy_name in policy_names:
        # Initialize the top-level dictionaries
        train_attack_results[policy_name] = defaultdict(list)
        test_attack_results[policy_name] = defaultdict(list)

        for rate in rates:
            train_policy_name = policy_name if args.train_policy is None else args.train_policy

            # Get the results from training and validation
            val_levels = train_log['val'][train_policy_name][rate][0]['output_levels']
            val_preds = train_log['val'][train_policy_name][rate][0]['preds']

            test_levels = eval_log['test'][policy_name][rate][0]['output_levels']
            test_preds = eval_log['test'][policy_name][rate][0]['preds']

            rand = np.random.RandomState(seed=5234)

            # Build the attack datasets
            train_attack_inputs, train_attack_outputs = make_noisy_dataset(levels=val_levels,
                                                                           preds=val_preds,
                                                                           window_size=window_size,
                                                                           num_trials=num_trials,
                                                                           rand=rand,
                                                                           noise_rate=noise_rate)

            test_attack_inputs, test_attack_outputs = make_noisy_dataset(levels=test_levels,
                                                                         preds=test_preds,
                                                                         window_size=window_size,
                                                                         num_trials=num_trials,
                                                                         rand=rand,
                                                                         noise_rate=noise_rate)

            # Evaluate the most-frequent classifier
            train_acc = most_freq_clf.score(train_attack_inputs, train_attack_outputs)
            test_acc = most_freq_clf.score(test_attack_inputs, test_attack_outputs)

            train_attack_results[policy_name][most_freq_clf.name].append(train_acc)
            test_attack_results[policy_name][most_freq_clf.name].append(test_acc)

            # Fit and evaluate the majority attack classifier
            majority_clf = MajorityClassifier()
            majority_clf.fit(train_attack_inputs, train_attack_outputs)

            train_acc = majority_clf.score(train_attack_inputs, train_attack_outputs)
            test_acc = majority_clf.score(test_attack_inputs, test_attack_outputs)

            train_attack_results[policy_name][majority_clf.name].append(train_acc)
            test_attack_results[policy_name][majority_clf.name].append(test_acc)

            # Fit and evaluate the decision tree ensemble classifier
            lr_clf = LogisticRegressionClassifier()
            lr_clf.fit(train_attack_inputs, train_attack_outputs)

            train_acc = lr_clf.score(train_attack_inputs, train_attack_outputs)
            test_acc = lr_clf.score(test_attack_inputs, test_attack_outputs)

            train_attack_results[policy_name][lr_clf.name].append(train_acc)
            test_attack_results[policy_name][lr_clf.name].append(test_acc)

    # Save the results
    eval_log['attack_test'] = test_attack_results
    eval_log['attack_train'] = train_attack_results
    eval_log['attack_train_log'] = train_log_path
    save_json_gz(eval_log, eval_log_path)
