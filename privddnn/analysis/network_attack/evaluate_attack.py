"""
Computes the accuracy in terms of recovering exit decisions from a packet capture.
"""
from argparse import ArgumentParser
from privddnn.utils.file_utils import read_jsonl_gz
from privddnn.utils.metrics import accuracy_score
from privddnn.attack.attack_predictions import attack_predictions, get_majority_preds
from privddnn.attack.attack_classifiers import LogisticRegressionCount


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--recovered-file', type=str, required=True, help='Path to the jsonl gz file of recovered decisions from the path capture.')
    parser.add_argument('--true-file', type=str, required=True, help='Path to the true results jsonl gz file.')
    parser.add_argument('--attack-model', type=str, required=True, help='Path to the serialized attack model.')
    args = parser.parse_args()

    # Load the results
    recovered = list(read_jsonl_gz(args.recovered_file))
    ground_truth = list(read_jsonl_gz(args.true_file))

    # Check the recovery rate on the exit decisions
    rec_exit_decisions = [record['exit_decision'] for record in recovered]
    true_exit_decisions = [record['exit_decision'] for record in ground_truth]

    for i in range(len(rec_exit_decisions)):
        if rec_exit_decisions[i] != true_exit_decisions[i]:
            print('Index: {}, Time: {}'.format(i, recovered[i]['time']))

    exit_decision_accuracy = accuracy_score(rec_exit_decisions, true_exit_decisions)
    print('Exit Decision Accuracy: {:.4f}'.format(exit_decision_accuracy))

    # Restore the attacker model
    attack_model = LogisticRegressionCount.restore(args.attack_model)

    # Execute the attack model on moving windows
    attack_preds = attack_predictions(predicted_decisions=rec_exit_decisions,
                                      attack_model=attack_model,
                                      num_outputs=2)

    majority_preds = get_majority_preds(predictions=[record['prediction'] for record in ground_truth],
                                        window_size=attack_model.window_size)

    attack_accuracy = accuracy_score(attack_preds, majority_preds)
    print('Attacker Accuracy: {:.4f}'.format(attack_accuracy))
