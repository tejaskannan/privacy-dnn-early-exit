import tensorflow as tf2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Tuple

from neural_network.branchynet_cnn import BranchyNetCNN
from neural_network.branchynet_dnn import BranchyNetDNN
from exiting.early_exit import random_exit, entropy_exit, max_prob_exit
from utils.metrics import compute_accuracy, compute_mutual_info


def execute_for_rate(probs: np.ndarray, rate: float, labels: np.ndarray, use_max_probs: bool, rand: np.random.RandomState) -> Tuple[float, float]:
    # Get the batch result
    if use_max_probs:
        result = max_prob_exit(probs=model_probs, rates=[1.0 - rate, rate])
    else:
        result = random_exit(probs=model_probs, rates=[1.0 - rate, rate], rand=rand)

    accuracy = compute_accuracy(result.preds, labels=labels)
    mutual_information = compute_mutual_info(result.output_counts, labels)

    return accuracy, mutual_information


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--output-path', type=str, help='Path to save the final plot')
    args = parser.parse_args()

    # Restore the model
    model = BranchyNetCNN.restore(path=args.model_path)

    # Get the dataset
    _, (test_inputs, test_labels) = tf2.keras.datasets.cifar10.load_data()
    test_labels = test_labels.reshape(-1)

    # Get the predictions from both models
    model_probs = model.predict(inputs=test_inputs, return_probs=True)  # [B, K]

    rates = np.arange(0.1, 1.01, 0.1)
    rand = np.random.RandomState(seed=591)

    rand_accuracy: List[float] = []
    rand_info: List[float] = []
    max_accuracy: List[float] = []
    max_info: List[float] = []

    for rate in rates:
        rand_acc, rand_mi = execute_for_rate(model_probs, rate=rate, labels=test_labels, use_max_probs=False, rand=rand)
        max_acc, max_mi = execute_for_rate(model_probs, rate=rate, labels=test_labels, use_max_probs=True, rand=rand)

        rand_accuracy.append(rand_acc)
        rand_info.append(rand_mi)
        max_accuracy.append(max_acc)
        max_info.append(max_mi)

    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2) = plt.subplots(figsize=(6, 4), nrows=1, ncols=2)

        ax1.plot(rates, rand_accuracy, label='Random', linewidth=3, marker='o', markersize=8)
        ax1.plot(rates, max_accuracy, label='Max Prob', linewidth=3, marker='o', markersize=8)

        ax2.plot(rates, rand_info, label='Random', linewidth=3, marker='o', markersize=8)
        ax2.plot(rates, rand_accuracy, label='Max Prob', linewidth=3, marker='o', markersize=8)

        ax1.legend()
        ax1.set_xlabel('Frac Stopping at 1st Output')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')

        ax2.legend()
        ax2.set_xlabel('Frac Stopping at 1st Output')
        ax2.set_ylabel('Mutual Information')
        ax2.set_title('Output vs Label Mut Inf')

        plt.tight_layout()

        if args.output_path is None:
            plt.show()
        else:
            plt.savefig(args.output_path, bbox_inches='tight', transparent=True)
