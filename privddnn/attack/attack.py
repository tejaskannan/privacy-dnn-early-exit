import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from typing import DefaultDict, List, Tuple

from privddnn.dataset.dataset import Dataset
from privddnn.utils.file_utils import read_json_gz, save_json_gz


def make_attack_dataset(outputs: np.ndarray, labels: np.ndarray, window_size: int, num_samples: int, rand: np.random.RandomState, noise_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    output_dist: DefaultDict[int, List[int]] = defaultdict(list)

    for num_outputs, label in zip(outputs, labels):
        output_dist[label].append(num_outputs)

    num_labels = len(output_dist)
    samples_per_label = int(num_samples / num_labels)

    input_list: List[np.ndarray] = []
    output_list: List[np.ndarray] = []

    num_noise = int(noise_rate * window_size)

    for label, output_counts in output_dist.items():
        for _ in range(samples_per_label):

            selected_counts = rand.choice(output_counts, size=window_size - num_noise)
            noise_counts = rand.choice(outputs, size=num_noise)

            # Create the features
            count = np.sum(selected_counts)
            count += np.sum(noise_counts)

            input_list.append(count)
            output_list.append(label)

    return np.vstack(input_list).reshape(-1), np.vstack(output_list).reshape(-1)


class MajorityClassifier:

    def __init__(self):
        self._clf: Dict[int, int] = dict()
        self._most_freq = 0

    def fit(self, inputs: np.ndarray, labels: np.ndarray):
        label_counts: DefaultDict[int, List[int]] = defaultdict(list)
        num_labels = np.max(labels) + 1

        for count, label in zip(inputs, labels):
            label_counts[count].append(label)

        for count, count_labels in sorted(label_counts.items()):
            freq = np.bincount(count_labels, minlength=num_labels)
            most_freq = np.argmax(freq)
            self._clf[count] = most_freq

        label_counts = np.bincount(labels, minlength=num_labels)
        self._most_freq = np.argmax(label_counts)

    def predict(self, count: int) -> int:
        return self._clf.get(count, self._most_freq)

    def score(self, inputs: np.ndarray, labels: np.ndarray) -> float:
        correct_count = 0.0
        total_count = 0.0

        for count, label in zip(inputs, labels):
            pred = self.predict(count)
            correct_count += float(pred == label)
            total_count += 1.0

        return correct_count / total_count


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    args = parser.parse_args()

    # Read the test log
    test_log = read_json_gz(args.test_log)

    # Read the validation and testing labels
    dataset_name = args.test_log.split(os.sep)[-3]
    dataset = Dataset(dataset_name)
    val_labels = dataset.get_val_labels()
    test_labels = dataset.get_test_labels()

    rates = [str(round(r / 10.0, 2)) for r in range(11)]
    #rates = ['0.7']
    policy_names = ['random', 'max_prob', 'label_max_prob', 'optimized_max_prob']
    window_size = 25
    noise_rate = 0.2
    num_samples = 2000

    train_attack_results: DefaultDict[str, List[float]] = defaultdict(list)
    test_attack_results: DefaultDict[str, List[float]] = defaultdict(list)

    for policy_name in policy_names:
        for rate in rates:
            val_outputs = test_log['val'][policy_name][rate]['output_levels']
            test_outputs = test_log['test'][policy_name][rate]['output_levels']
            rand = np.random.RandomState(seed=5234)

            train_attack_inputs, train_attack_outputs = make_attack_dataset(outputs=val_outputs,
                                                                            labels=val_labels,
                                                                            window_size=window_size,
                                                                            num_samples=num_samples,
                                                                            rand=rand,
                                                                            noise_rate=noise_rate)

            test_attack_inputs, test_attack_outputs = make_attack_dataset(outputs=test_outputs,
                                                                          labels=test_labels,
                                                                          window_size=window_size,
                                                                          num_samples=num_samples,
                                                                          rand=rand,
                                                                          noise_rate=noise_rate)

            # Fit the model
            clf = MajorityClassifier()
            clf.fit(train_attack_inputs, train_attack_outputs)

            train_acc = clf.score(train_attack_inputs, train_attack_outputs)
            test_acc = clf.score(test_attack_inputs, test_attack_outputs)

            train_attack_results[policy_name].append(train_acc)
            test_attack_results[policy_name].append(test_acc)

    # Save the results
    test_log['attack_test'] = test_attack_results
    test_log['attack_train'] = train_attack_results
    save_json_gz(test_log, args.test_log)

    # Get the most frequent label
    #label_counts = np.bincount(val_outputs, minlength=np.max(val_outputs) + 1)
    #most_freq_label = np.argmax(label_counts)

    #most_freq_accuracy = np.average((test_attack_outputs == most_freq_label).astype(float))
    #print('Most Freq: {:.5f}'.format(most_freq_accuracy))

    # TODO: Integrate Top 2 Accuracy, F1 Score
    # TODO: Serialize the result


    #with plt.style.context('seaborn-ticks'):
    #    fig, ax = plt.subplots()

    #    xs: DefaultDict[int, List[float]] = defaultdict(list)
    #    ys: DefaultDict[int, List[float]] = defaultdict(list)

    #    for input_vector, label in zip(attack_inputs, attack_outputs):
    #        xs[label].append(input_vector[0])
    #        ys[label].append(input_vector[1])

    #    print(len(xs))
    #    print(len(xs[0]))

    #    markers = ['o', 'v', '^', 'P', 's', '*', 'X', 'H', 'D', 'p']

    #    for idx, label in enumerate(sorted(xs.keys())):
    #        ax.scatter(xs[label], ys[label], label=label, marker=markers[idx])

    #    plt.show()
