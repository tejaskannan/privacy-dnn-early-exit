import numpy as np
from argparse import ArgumentParser
from collections import Counter
from typing import Tuple

from privddnn.utils.file_utils import read_json_gz


def get_most_freq_for_level(preds: np.ndarray, levels: np.ndarray, target_level: int) -> Tuple[int, float]:
    total_count = 0
    pred_counter: Counter = Counter()

    for pred, level in zip(preds, levels):
        if level == target_level:
            pred_counter[pred] += 1
            total_count += 1

    if total_count == 0:
        return 0, 0.0

    most_freq_pred, pred_count = pred_counter.most_common(1)[0]
    return most_freq_pred, pred_count / total_count


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True, help='Path to the test log.')
    parser.add_argument('--policy', type=str, required=True, help='Name of the policy to analyze.')
    parser.add_argument('--rate', type=str, required=True, help='The exit rate to analyze.')
    args = parser.parse_args()

    val_log = read_json_gz(args.test_log)['val']
    policy_log = val_log[args.policy][args.rate]

    preds = np.array(policy_log['randomized']['preds'])
    levels = np.array(policy_log['randomized']['output_levels'])

    # Get the most frequent prediction overall
    pred_counts = np.bincount(preds).astype(float)
    pred_freq = pred_counts / np.sum(pred_counts)
    max_idx = np.argmax(pred_freq)
    print('Most Freq Overall: {} ({:4f})'.format(max_idx, pred_freq[max_idx] * 100))

    # Get the most frequent predictions stratified by level
    most_freq_pred, pred_freq = get_most_freq_for_level(preds, levels, target_level=0)
    print('Most Freq Level 0: {} ({:4f})'.format(most_freq_pred, pred_freq * 100))

    most_freq_pred, pred_freq = get_most_freq_for_level(preds, levels, target_level=1)
    print('Most Freq Level 1: {} ({:4f})'.format(most_freq_pred, pred_freq * 100))
