import numpy as np
from argparse import ArgumentParser
from collections import Counter
from typing import List

from privddnn.analysis.msp_comp_energy import get_energy_per_period, get_exit_points
from privddnn.attack.attack_classifiers import LogisticRegressionCount
from privddnn.utils.file_utils import read_jsonl_gz
from privddnn.utils.metrics import accuracy_score


def attack(predicted_decisions: List[int], attack_model: LogisticRegressionCount, window_size: int, num_outputs: int, num_labels: int):
    attack_preds: List[int] = []
    for window_start in range(0, len(predicted_decisions), window_size):
        window_end = window_start + args.window_size
        window_decisions = predicted_decisions[window_start:window_end]

        # Turn decisions into one-hot encodings
        features: List[np.ndarray] = []
        for decision in window_decisions:
            feature_vector = np.zeros(shape=(num_outputs, ), dtype=float)
            feature_vector[decision] = 1.0
            features.append(np.expand_dims(feature_vector, axis=0))

        input_features = np.vstack(features)  # [W, D]
        predictions, _ = attack_model.predict_rankings(inputs=input_features, top_k=1)
        attack_preds.append(predictions[0])

    return attack_preds


def get_majority_preds(predictions: List[int], window_size: int) -> List[int]:
    majority_list: List[int] = []

    for window_start in range(0, len(predictions), window_size):
        window_end = window_start + window_size

        pred_counter: Counter = Counter()
        for pred in predictions[window_start:window_end]:
            pred_counter[pred] += 1

        majority = pred_counter.most_common(1)[0][0]
        majority_list.append(majority)

    return majority_list


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--energy-file', type=str, required=True)
    parser.add_argument('--preds-file', type=str, required=True)
    parser.add_argument('--attack-model-file', type=str, required=True)
    parser.add_argument('--window-size', type=int, required=True)
    parser.add_argument('--num-labels', type=int, required=True)
    args = parser.parse_args()

    assert args.window_size >= 1, 'Must provide a positive window size.'

    # Read the existing results
    true_decisions: List[int] = []
    true_predictions: List[int] = []

    num_correct = 0
    total_count = 0

    for record in read_jsonl_gz(args.preds_file):
        true_decisions.append(int(record['exit_decision']))
        true_predictions.append(int(record['prediction']))
        num_correct += int(int(record['prediction']) == int(record['label']))
        total_count += 1

    # TODO: Remove dependence on the number of samples
    num_outputs = max(true_decisions) + 1
    num_samples = len(true_decisions)

    # Get the energy per period and the predicted decisions
    energy_per_period = get_energy_per_period(path=args.energy_file, output_file=None, should_plot=False)[:num_samples]
    predicted_decisions = get_exit_points(energy=energy_per_period, num_outputs=num_outputs)

    # Compute the recovery accuracy on the exit decisions
    recovery_accuracy = accuracy_score(true_decisions, predicted_decisions)

    # Compute the true exit rates
    true_exit_rates = np.bincount(true_decisions, minlength=num_outputs).astype(float)
    true_exit_rates /= float(total_count)

    print('Number of outputs: {}, Number of Samples: {}'.format(num_outputs, num_samples))
    print('Recovery Accuracy: {:.4f}'.format(recovery_accuracy))
    print('Inference Accuracy: {:.4f} ({} / {})'.format(num_correct / total_count, num_correct, total_count))
    print('True Exit Decisions: {}'.format(true_exit_rates))

    # Add the attack accuracy by loading a serialized attack model
    attack_model = LogisticRegressionCount.restore(args.attack_model_file, window_size=args.window_size, num_labels=args.num_labels)

    attack_preds = attack(predicted_decisions=predicted_decisions,
                          attack_model=attack_model,
                          num_outputs=num_outputs,
                          window_size=args.window_size,
                          num_labels=args.num_labels)

    majority_true_preds = get_majority_preds(true_predictions, args.window_size)
    attack_accuracy = accuracy_score(majority_true_preds, attack_preds)
    
    attack_correct = np.sum(np.equal(majority_true_preds, attack_preds).astype(int))
    total_count = len(attack_preds)

    print('Attack Accuracy: {:.4f} ({} / {})'.format(attack_accuracy, attack_correct, total_count))
