import csv
import numpy as np
from argparse import ArgumentParser
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from typing import List, Tuple

from privddnn.attack.attack_classifiers import ACCURACY, DecisionTreeEnsembleCount
from privddnn.utils.file_utils import read_jsonl_gz


MAC_ADDRESS = 'A0:6C:65:CF:81:D4'
PROTOCOL = 'ATT'
TIME_THRESHOLD = 0.5
DROP_RATE = 0.05

SIZE_CUTOFF= 100


def extract_message_sizes(path: str, rand: np.random.RandomState) -> Tuple[List[int], int, int]:
    prev_time = None
    current_size = 0
    result_times: List[float] = []

    start_time = None
    end_time = None

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

                if start_time is None:
                    start_time = prev_time

                if current_size > SIZE_CUTOFF:
                    gaps.append(packet_time - prev_time)

                    result_times.append(prev_time)
               
                prev_time = packet_time
                current_size = 0

            current_size += packet_length

    if (current_size > SIZE_CUTOFF):
        result_times.append(prev_time)

    end_time = prev_time

    return result_times, start_time, end_time


#def classify_decisions(message_sizes: List[int]) -> List[int]:
#    clf = KMeans(n_clusters=2)
#    clusters = clf.fit_predict(np.array(message_sizes).reshape(-1, 1))
#
#    # Rename the clusters based on sizes
#    centers = clf.cluster_centers_
#    max_idx = int(centers[0] < centers[1])
#
#    return [int(c == max_idx) for c in clusters]


def classify_decisions(result_times: List[int], start_time: int, end_time: int, period: float):
    exit_decisions: List[int] = []

    # Handle the start
    gap = result_times[0] - start_time
    num_early = max(int(gap / period), 0)

    for _ in range(num_early):
        exit_decisions.append(0)

    exit_decisions.append(1)

    # Handle the gaps between result times
    for idx in range(1, len(result_times)):
        gap = result_times[idx] - result_times[idx - 1]
        
        num_early = max(int(gap / period) - 1, 0)
        for _ in range(num_early):
            exit_decisions.append(0)

        exit_decisions.append(1)

    # Handle the end
    gap = end_time - result_times[-1]
    num_early = max(int(gap / period), 0)

    for _ in range(num_early):
        exit_decisions.append(0)

    return exit_decisions


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trace-path', type=str, required=True, help='Path to the Wireshark trace CSV.')
    parser.add_argument('--attack-model-path', type=str, required=True, help='Path to the saved attack model.')
    parser.add_argument('--log-path', type=str, required=True, help='Path to the recorded log.')
    args = parser.parse_args()

    rand = np.random.RandomState(2523)
    message_times, start_time, end_time = extract_message_sizes(args.trace_path, rand=rand)
    print('Number of Messages: {}'.format(len(message_times)))

    exit_decisions = classify_decisions(message_times, start_time, end_time, period=1.2)

    server_results = list(read_jsonl_gz(args.log_path))
    true_exit_decisions = [int(record['message_size'] == 128) for record in server_results]

    # Restore the attack model
    clf = DecisionTreeEnsembleCount.restore(args.attack_model_path)

    model_preds: List[int] = [int(record['pred']) for record in server_results]
    num_labels = clf.num_labels

    exit_decision_blocks: List[np.ndarray] = []
    labels: List[int] = []

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
