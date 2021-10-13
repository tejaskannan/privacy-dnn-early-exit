import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from typing import DefaultDict, List, Tuple, Dict

from privddnn.utils.file_utils import read_json_gz, save_json_gz
from privddnn.attack.nearest_index import make_similar_attack_dataset


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
        self._clf: Dict[int, List[int]] = dict()
        self._most_freq = 0

    def fit(self, inputs: np.ndarray, labels: np.ndarray):
        label_counts: DefaultDict[int, List[int]] = defaultdict(list)
        num_labels = np.max(labels) + 1

        for count, label in zip(inputs, labels):
            label_counts[count].append(label)

        for count, count_labels in sorted(label_counts.items()):
            freq = np.bincount(count_labels, minlength=num_labels)
            most_freq = np.argsort(freq)[::-1]
            self._clf[count] = most_freq

        label_counts = np.bincount(labels, minlength=num_labels)
        self._most_freq = np.argsort(label_counts)[::-1]

    def predict(self, count: int) -> int:
        return self.predict_rankings(count=count, top_k=1)[0]

    def predict_rankings(self, count: int, top_k: int) -> List[int]:
        rankings = self._clf.get(count, self._most_freq)
        return rankings[0:top_k].astype(int).tolist()

    def score(self, inputs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        correct_count = 0.0
        top2_count = 0.0
        total_count = 0.0

        for count, label in zip(inputs, labels):
            preds = self.predict_rankings(count, top_k=2)

            top2_count += float(label in preds)
            correct_count += float(preds[0] == label)
            total_count += 1.0

        return {
            'accuracy': correct_count / total_count,
            'top2': top2_count / total_count
        }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train-log', type=str, required=True)
    parser.add_argument('--eval-log', type=str, required=True)
    parser.add_argument('--train-policy', type=str, choices=['max_prob', 'entropy'])
    args = parser.parse_args()

    # Read the logs for training and testing
    train_log = read_json_gz(args.train_log)
    eval_log = read_json_gz(args.eval_log)

    # Get paths to the 'similarity' indices
    dataset_name = args.train_log.split(os.sep)[-3]
    val_index_path = '../data/{}/val'.format(dataset_name)
    test_index_path = '../data/{}/test'.format(dataset_name)

    rates = [str(round(r / 20.0, 2)) for r in range(21)]
    #policy_names = ['random', 'max_prob', 'label_max_prob', 'hybrid_max_prob', 'entropy', 'label_entropy', 'hybrid_entropy']
    policy_names = ['random', 'max_prob', 'entropy']

    window_size = 25
    noise_rate = 0.2
    #num_samples = 2000
    num_samples = 1200

    train_attack_results: DefaultDict[str, List[float]] = defaultdict(list)
    test_attack_results: DefaultDict[str, List[float]] = defaultdict(list)

    for policy_name in policy_names:
        for rate in rates:
            train_policy_name = policy_name if args.train_policy is None else args.train_policy

            val_outputs = train_log['val'][train_policy_name][rate][0]['output_levels']
            val_preds = train_log['val'][train_policy_name][rate][0]['preds']

            test_outputs = eval_log['test'][policy_name][rate][0]['output_levels']
            test_preds = eval_log['test'][policy_name][rate][0]['preds']

            rand = np.random.RandomState(seed=5234)

            train_attack_inputs, train_attack_outputs = make_attack_dataset(outputs=val_outputs,
                                                                            labels=val_preds,
                                                                            window_size=window_size,
                                                                            num_samples=num_samples,
                                                                            rand=rand,
                                                                            noise_rate=noise_rate)

            test_attack_inputs, test_attack_outputs = make_attack_dataset(outputs=test_outputs,
                                                                          labels=test_preds,
                                                                          window_size=window_size,
                                                                          num_samples=num_samples,
                                                                          rand=rand,
                                                                          noise_rate=noise_rate)

            #train_attack_inputs, train_attack_outputs = make_similar_attack_dataset(levels=val_outputs,
            #                                                                labels=val_preds,
            #                                                                window_size=window_size,
            #                                                                num_samples=num_samples,
            #                                                                rand=rand,
            #                                                                path=val_index_path)

            #test_attack_inputs, test_attack_outputs = make_similar_attack_dataset(levels=test_outputs,
            #                                                              labels=test_preds,
            #                                                              window_size=window_size,
            #                                                              num_samples=num_samples,
            #                                                              rand=rand,
            #                                                              path=test_index_path)


            # Fit the model
            clf = MajorityClassifier()
            clf.fit(train_attack_inputs, train_attack_outputs)

            train_acc = clf.score(train_attack_inputs, train_attack_outputs)
            test_acc = clf.score(test_attack_inputs, test_attack_outputs)

            train_attack_results[policy_name].append(train_acc)
            test_attack_results[policy_name].append(test_acc)

    # Create the 'Most Frequent' dataset that uses the most frequent label from the test model
    #val_counts = np.array(train_log['val']['most_freq']).astype(float)  # [2, L] array
    #val_freq = val_counts / np.sum(val_counts, axis=-1, keepdims=True)

    #test_counts = np.array(eval_log['test']['most_freq']).astype(float)  # [2, L] array
    #test_freq = test_counts / np.sum(test_counts, axis=-1, keepdims=True)

    #for rate in rates:
    #    r = float(rate)
    #    weighted_val_freq = val_freq[0, :] * r + (1.0 - r) * val_freq[1, :]
    #    most_freq = np.argmax(weighted_val_freq)

    #    val_most_freq_accuracy = (val_counts[0, most_freq] / np.sum(val_counts[0])) * r + (val_counts[1, most_freq] / np.sum(val_counts[1])) * (1.0 - r)
    #    test_most_freq_accuracy = (test_counts[0, most_freq] / np.sum(test_counts[0])) * r + (test_counts[1, most_freq] / np.sum(test_counts[1])) * (1.0 - r)

    #    train_attack_results['most_freq'].append(val_most_freq_accuracy)
    #    test_attack_results['most_freq'].append(test_most_freq_accuracy)

    # Save the results
    eval_log['attack_test'] = test_attack_results
    eval_log['attack_train'] = train_attack_results
    save_json_gz(eval_log, args.eval_log)

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
