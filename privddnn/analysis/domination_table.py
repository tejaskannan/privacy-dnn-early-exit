import numpy as np
from argparse import ArgumentParser
from typing import List, Dict
from privddnn.utils.file_utils import read_json_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    args = parser.parse_args()

    test_log = read_json_gz(args.test_log)['test']
    accuracy_scores: Dict[str, Dict[str, float]] = dict()

    for policy_name, policy_results in sorted(test_log.items()):
        policy_scores: Dict[str, float] = dict()

        for rate, rate_results in policy_results.items():
            preds = rate_results[args.dataset_order]['preds']
            labels = rate_results[args.dataset_order]['labels']
            accuracy = np.average(np.isclose(preds, labels).astype(float))
            policy_scores[rate] = accuracy
            
        accuracy_scores[policy_name] = policy_scores

    
    print('& {}'.format(' & '.join(sorted(accuracy_scores.keys()))))

    for first_policy in sorted(accuracy_scores.keys()):
        print('{} &'.format(first_policy), end=' ')
        comparison: List[float] = []

        for second_policy in sorted(accuracy_scores.keys()):

            num_greater = 0
            total_rates = 0
            for rate in accuracy_scores[first_policy].keys():
                if np.isclose(float(rate), 0.0) or np.isclose(float(rate), 1.0):
                    continue

                first_accuracy = accuracy_scores[first_policy][rate]
                second_accuracy = accuracy_scores[second_policy][rate]
                num_greater += int(first_accuracy > second_accuracy)
                total_rates += 1

            comparison.append((num_greater, total_rates, num_greater / total_rates))

        print(' & '.join(map(lambda t: '{} / {} ({:.4f})'.format(t[0], t[1], t[2]), comparison)))
