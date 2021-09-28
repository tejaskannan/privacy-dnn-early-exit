import tensorflow as tf2
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum, auto
from typing import List, Tuple, DefaultDict

from neural_network import restore_model, NeuralNetwork, OpName, ModelMode
from exiting.early_exit import entropy_exit
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--output-path', type=str, help='Path to save the final plot')
    args = parser.parse_args()

    # Restore the model
    model: NeuralNetwork = restore_model(path=args.model_path, model_mode=ModelMode.TEST)

    # Get the predictions from the models
    model_preds = model.test(op=OpName.PREDICTIONS)  # [B, K]
    stop_probs = model.test(op=OpName.STOP_PROBS)  # [B, K]
    test_labels = model.dataset.get_test_labels()

    num_correct = 0.0
    num_total = 0.0
    output_indices = defaultdict(list)

    rand = np.random.RandomState(seed=7894)
    for idx in range(len(test_labels)):
        label = test_labels[idx]
        stop_rates = stop_probs[idx]

        if idx < 10:
            print('Label: {}, Stop Rates: {}'.format(label, stop_rates))

        r = rand.uniform()
        stop_idx = int(r > stop_rates[0])

        pred = model_preds[idx, stop_idx]
        num_correct += int(np.isclose(pred, label))
        num_total += 1.0

        output_indices[label].append(stop_idx)

    for label, stops in sorted(output_indices.items()):
        avg = np.average(stops)
        std = np.std(stops)
        print('{} -> {}, {}'.format(label, avg, std))

    print('Accuracy: {}'.format(num_correct / num_total))
