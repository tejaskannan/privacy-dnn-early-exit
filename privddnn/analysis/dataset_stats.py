import numpy as np
from argparse import ArgumentParser
from collections import defaultdict, namedtuple, Counter
from typing import Dict

from privddnn.dataset import Dataset, DataFold
from privddnn.dataset.data_iterators import DataIterator, make_data_iterator


DataStats = namedtuple('DataStats', ['distribution', 'window_majority', 'most_common_freq'])


def print_distribution(distribution: Dict[int, float]):
    values = list(distribution.values())
    print('Max Fraction: {:.5f}'.format(np.amax(values)))

    #for label, fraction in sorted(distribution.items()):
    #    print('\t{}: {:.5f}'.format(label, fraction))


def get_label_stats(iterator: DataIterator, window_size: int) -> DataStats:
    label_counts: Counter = Counter()
    majority_counts: Counter = Counter()
    total_count = 0
    num_windows = 0

    label_window: List[int] = []
    majority_fractions: List[float] = []

    for idx, (sample, probs, label) in enumerate(iterator):
        label_counts[label] += 1
        total_count += 1
        label_window.append(label)

        if len(label_window) == window_size:
            window_counter: Counter = Counter()

            for label in label_window:
                window_counter[label] += 1

            most_common = window_counter.most_common(1)[0]
            majority_frac = most_common[1] / window_size
            majority_fractions.append(majority_frac)
            majority_counts[most_common[0]] += 1
            num_windows += 1

            label_window = []

    label_distribution: Dict[int, float] = dict()
    for label, count in label_counts.items():
        label_distribution[label] = count / total_count

    (most_common, most_common_count) = majority_counts.most_common(1)[0]

    return DataStats(distribution=label_distribution,
                     most_common_freq=(most_common_count / total_count),
                     window_majority=np.average(majority_fractions))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--window-sizes', type=int, nargs='+', required=True)
    args = parser.parse_args()

    dataset = Dataset(dataset_name=args.dataset_name)

    # Get the number of validation, training, and testing samples
    print('# Train: {}'.format(dataset.get_train_inputs().shape))
    print('# Val: {}'.format(dataset.get_val_inputs().shape))
    print('# Test: {}'.format(dataset.get_test_inputs().shape))
    print('# Labels: {}'.format(np.amax(dataset.get_train_labels()) + 1))

    # Get statistics for each window size
    for window_size in args.window_sizes:
        print('======== {} ========'.format(window_size))

        val_iterator = make_data_iterator(name='same-label',
                                          dataset=dataset,
                                          window_size=window_size,
                                          pred_probs=None,
                                          num_reps=1,
                                          fold='val')
        val_stats = get_label_stats(iterator=val_iterator, window_size=window_size)

        test_iterator = make_data_iterator(name='same-label',
                                           dataset=dataset,
                                           window_size=window_size,
                                           pred_probs=None,
                                           num_reps=1,
                                           fold='test')
        test_stats = get_label_stats(iterator=test_iterator, window_size=window_size)

        print('Val Avg Majority Fraction: {:.5f}'.format(val_stats.window_majority))
        print('Test Avg Majority Fraction: {:.5f}'.format(test_stats.window_majority))

        print('Val Most Freq Proportion: {:.5f}'.format(val_stats.most_common_freq))
        print('Test Most Freq Proportion: {:.5f}'.format(val_stats.most_common_freq))

        print('Val Distribution')
        print_distribution(distribution=val_stats.distribution)

        print('Test Distribution')
        print_distribution(distribution=test_stats.distribution)
