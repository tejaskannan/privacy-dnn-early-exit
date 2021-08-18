import tensorflow as tf2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum, auto
from typing import List, Tuple, DefaultDict

from neural_network.branchynet_cnn import BranchyNetCNN
from neural_network.branchynet_dnn import BranchyNetDNN
from exiting.early_exit import random_exit, entropy_exit, max_prob_exit
from utils.metrics import compute_accuracy, compute_mutual_info


class ExitStrategy(Enum):
    RANDOM = auto()
    MAX_PROB = auto()
    ENTROPY = auto()


def execute_for_rate(probs: np.ndarray, rate: float, labels: np.ndarray, strategy: ExitStrategy, rand: np.random.RandomState) -> Tuple[float, float]:
    # Get the batch result
    if strategy == ExitStrategy.MAX_PROB:
        result = max_prob_exit(probs=model_probs, rates=[1.0 - rate, rate])
    elif strategy == ExitStrategy.ENTROPY:
        result = entropy_exit(probs=model_probs, rates=[1.0 - rate, rate])
    elif strategy == ExitStrategy.RANDOM:
        result = random_exit(probs=model_probs, rates=[1.0 - rate, rate], rand=rand)
    else:
        raise ValueError('Unknown exit strategy: {}'.format(strategy))

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

    # Get the predictions from the models
    model_probs = model.predict(inputs=test_inputs, return_probs=True)  # [B, K]

    rates = np.arange(0.1, 1.01, 0.1)
    rand = np.random.RandomState(seed=591)

    # Execute all early stopping policies
    accuracy_dict: DefaultDict[ExitStrategy, List[float]] = defaultdict(list)
    info_dict: DefaultDict[ExitStrategy, List[float]] = defaultdict(list)

    for rate in rates:
        rand_acc, rand_mi = execute_for_rate(model_probs, rate=rate, labels=test_labels, strategy=ExitStrategy.RANDOM, rand=rand)
        max_acc, max_mi = execute_for_rate(model_probs, rate=rate, labels=test_labels, strategy=ExitStrategy.MAX_PROB, rand=rand)
        ent_acc, ent_mi = execute_for_rate(model_probs, rate=rate, labels=test_labels, strategy=ExitStrategy.ENTROPY, rand=rand)

        # Log the accuracy values
        accuracy_dict[ExitStrategy.RANDOM].append(rand_acc)
        accuracy_dict[ExitStrategy.MAX_PROB].append(max_acc)
        accuracy_dict[ExitStrategy.ENTROPY].append(ent_acc)

        # Log the information values
        info_dict[ExitStrategy.RANDOM].append(rand_mi)
        info_dict[ExitStrategy.MAX_PROB].append(max_mi)
        info_dict[ExitStrategy.ENTROPY].append(ent_mi)

    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2) = plt.subplots(figsize=(6, 4), nrows=1, ncols=2)

        ax1.plot(rates, accuracy_dict[ExitStrategy.RANDOM], label='Random', linewidth=3, marker='o', markersize=8, color='#8da0cb')
        ax1.plot(rates, accuracy_dict[ExitStrategy.MAX_PROB], label='Max Prob', linewidth=3, marker='o', markersize=8, color='#66c2a5')
        ax1.plot(rates, accuracy_dict[ExitStrategy.ENTROPY], label='Entropy', linewidth=3, marker='o', markersize=8, color='#fc8d62')

        ax2.plot(rates, info_dict[ExitStrategy.RANDOM], label='Random', linewidth=3, marker='o', markersize=8, color='#8da0cb')
        ax2.plot(rates, info_dict[ExitStrategy.MAX_PROB], label='Max Prob', linewidth=3, marker='o', markersize=8, color='#66c2a5')
        ax2.plot(rates, info_dict[ExitStrategy.ENTROPY], label='Entropy', linewidth=3, marker='o', markersize=8, color='#fc8d62')

        ax1.legend()
        ax1.set_xlabel('Frac Stopping at 1st Output')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')

        ax2.set_xlabel('Frac Stopping at 1st Output')
        ax2.set_ylabel('Mutual Information')
        ax2.set_title('Output vs Label Mut Inf')

        plt.tight_layout()

        if args.output_path is None:
            plt.show()
        else:
            plt.savefig(args.output_path, bbox_inches='tight', transparent=True)
