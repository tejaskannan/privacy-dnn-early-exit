import csv
import numpy as np
from argparse import ArgumentParser
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from typing import List

from privddnn.attack.attack_classifiers import ACCURACY, DecisionTreeEnsembleCount
from privddnn.utils.file_utils import read_jsonl_gz


MAC_ADDRESS = 'A0:6C:65:CF:81:D4'
PROTOCOL = 'ATT'
TIME_THRESHOLD = 0.5
DROP_RATE = 0.01


def extract_message_sizes(path: str, rand: np.random.RandomState) -> List[int]:
    prev_time = None
    current_size = 0
    result: List[int] = []
    result_times: List[float] = []

    gaps = []

    with open(path, 'r') as fin:
        reader = csv.reader(fin, delimiter=',', quotechar='"')
        for idx, line in enumerate(reader):
            if idx == 0:
                continue

            packet_time = float(line[1])
            source_mac = line[2].replace(' ()', '')
            protocol = line[4]
            packet_length = int(line[5])

            if (source_mac.upper() != MAC_ADDRESS) or (protocol != PROTOCOL):
                continue

            r = rand.uniform()
            if (r < DROP_RATE):
                continue

            if prev_time is None:
                prev_time = packet_time

            if (packet_time - prev_time) >= TIME_THRESHOLD:
                gaps.append(packet_time - prev_time)

                result.append(current_size)
                result_times.append(prev_time)
                prev_time = packet_time
                current_size = 0

            current_size += packet_length

    if current_size > 0:
        result.append(current_size)

    return result, result_times


def classify_decisions(message_sizes: List[int]) -> List[int]:
    clf = KMeans(n_clusters=2)
    clusters = clf.fit_predict(np.array(message_sizes).reshape(-1, 1))

    # Rename the clusters based on sizes
    centers = clf.cluster_centers_
    max_idx = int(centers[0] < centers[1])

    return [int(c == max_idx) for c in clusters]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trace-path', type=str, required=True, help='Path to the Wireshark trace CSV.')
    parser.add_argument('--attack-model-path', type=str, required=True, help='Path to the saved attack model.')
    parser.add_argument('--log-path', type=str, required=True, help='Path to the recorded log.')
    args = parser.parse_args()

    rand = np.random.RandomState(2523)
    message_sizes, message_times = extract_message_sizes(args.trace_path, rand=rand)
    print('Number of Messages: {}'.format(len(message_sizes)))

    exit_decisions = classify_decisions(message_sizes)

    # Apply the classifier
    clf = DecisionTreeEnsembleCount.restore(args.attack_model_path)

    exit_decision_blocks: List[np.ndarray] = []
    labels: List[int] = []

    server_results = list(read_jsonl_gz(args.log_path))
    model_preds: List[int] = [int(record['pred']) for record in server_results]
    num_labels = clf.num_labels

    for start_idx in range(0, len(exit_decisions), clf.window_size):
        end_idx = start_idx + clf.window_size

        # Make the input features
        decision_block = np.zeros(shape=(clf.window_size, 2))

        for offset in range(clf.window_size):
            exit_decision = exit_decisions[start_idx + offset]
            decision_block[offset, exit_decision] = 1

        exit_decision_blocks.append(np.expand_dims(decision_block, axis=0))

        pred_counts = np.bincount(model_preds[start_idx:end_idx], minlength=num_labels)
        labels.append(int(np.argmax(pred_counts)))

    attack_results = clf.score(inputs=np.vstack(exit_decision_blocks),
                               labels=np.vstack(labels).reshape(-1).astype(int))

    true_labels: List[int] = [int(record['label']) for record in server_results]
    inference_accuracy = accuracy_score(y_true=true_labels, y_pred=model_preds)

    true_decisions: List[int] = [int(record['message_size'] >= 80) for record in server_results]

    print('Exit Rate: {:.4f}'.format(1.0 - np.mean(true_decisions)))
    print('Attack Accuracy: {:.4f}%'.format(100.0 * attack_results[ACCURACY]))
    print('Inference Accuracy: {:.4f}%'.format(100.0 * inference_accuracy))
