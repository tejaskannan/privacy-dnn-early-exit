import os
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, DefaultDict

from privddnn.analysis.utils.read_logs import get_attack_results
from privddnn.attack.attack_classifiers import DECISION_TREE_COUNT, DECISION_TREE_NGRAM, LOGISTIC_REGRESSION_COUNT, LOGISTIC_REGRESSION_NGRAM



if __name__ == '__main__':
    parser = ArgumentParser('Script to summarize attack results in black-box settings.')
    parser.add_argument('--test-log-folder', type=str, required=True)
    parser.add_argument('--attack-model', type=str, required=True, choices=[DECISION_TREE_COUNT, DECISION_TREE_NGRAM, LOGISTIC_REGRESSION_COUNT, LOGISTIC_REGRESSION_NGRAM])
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--attack-log-folders', type=str, nargs='+', required=True)
    args = parser.parse_args()

    max_attack_accuracy: DefaultDict[str, List[float]] = defaultdict(list)  # Policy Name -> [ max acc per trial ]

    for attack_log_folder in args.attack_log_folders:
        # Read the attack accuracy
        attack_results = get_attack_results(folder_path=args.test_log_folder,
                                            fold='test',
                                            dataset_order=args.dataset_order,
                                            attack_train_log=attack_log_folder,
                                            attack_policy='same',
                                            metric='accuracy',
                                            attack_model=args.attack_model,
                                            target_pred=None,
                                            trials=None)

        for policy_name in attack_results.keys():
            max_metric = 0.0

            for rate_results in attack_results[policy_name].values():
                for result in rate_results:
                    max_metric = max(result, max_metric)

            max_attack_accuracy[policy_name].append(max_metric)
        
    for policy_name in sorted(max_attack_accuracy.keys()):
        accuracy_values = max_attack_accuracy[policy_name]
        avg, std = np.mean(accuracy_values), np.std(accuracy_values)
        print('{} & {:.4f} ({:.4f}) \\'.format(policy_name, avg, std))
