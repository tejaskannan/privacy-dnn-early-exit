"""
Computes the accuracy in terms of recovering exit decisions from a packet capture.
"""
from argparse import ArgumentParser
from typing import Tuple

from privddnn.utils.file_utils import read_jsonl_gz
from privddnn.utils.metrics import accuracy_score
from privddnn.attack.attack_predictions import attack_predictions, get_majority_preds
from privddnn.attack.attack_classifiers import LogisticRegressionCount


def evaluate_attack(recovered_file: str, true_file: str, attack_model_file: str) -> Tuple[float, float, float]:
     # Load the results
    recovered = list(read_jsonl_gz(recovered_file))
    ground_truth = list(read_jsonl_gz(true_file))

    # Check the recovery rate on the exit decisions
    rec_exit_decisions = [record['exit_decision'] for record in recovered]
    true_exit_decisions = [record['exit_decision'] for record in ground_truth]

    exit_decision_accuracy = accuracy_score(rec_exit_decisions, true_exit_decisions)

    # Restore the attacker model
    attack_model = LogisticRegressionCount.restore(attack_model_file, window_size=None, num_labels=None)

    # Execute the attack model on moving windows
    attack_preds = attack_predictions(predicted_decisions=rec_exit_decisions,
                                      attack_model=attack_model,
                                      num_outputs=2)

    majority_preds = get_majority_preds(predictions=[record['prediction'] for record in ground_truth],
                                        window_size=attack_model.window_size)

    attack_accuracy = accuracy_score(attack_preds, majority_preds)

    # Get the inference accuracy (for good measure)
    model_preds = [record['prediction'] for record in ground_truth]
    labels = [record['label'] for record in ground_truth]
    inference_accuracy = accuracy_score(model_preds, labels)

    return exit_decision_accuracy, attack_accuracy, inference_accuracy


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--recovered-file', type=str, required=True, help='Path to the jsonl gz file of recovered decisions from the path capture.')
    parser.add_argument('--true-file', type=str, required=True, help='Path to the true results jsonl gz file.')
    parser.add_argument('--attack-model', type=str, required=True, help='Path to the serialized attack model.')
    args = parser.parse_args()

    exit_accuracy, attack_accuracy, inference_accuracy = evaluate_attack(recovered_file=args.recovered_file,
                                                                         true_file=args.true_file,
                                                                         attack_model_file=args.attack_model)
