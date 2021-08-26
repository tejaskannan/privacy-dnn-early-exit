import tensorflow as tf2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum, auto
from typing import List, Tuple, DefaultDict

from neural_network import restore_model, NeuralNetwork, OpName, ModelMode
from exiting.early_exit import random_exit, entropy_exit, max_prob_exit, even_max_prob_exit
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info


class ExitStrategy(Enum):
    RANDOM = auto()
    MAX_PROB = auto()
    ENTROPY = auto()
    EVEN_MAX_PROB = auto()
    EVEN_MAX_PROB_RAND = auto()


COLORS = {
    ExitStrategy.RANDOM: '#8da0cb',
    ExitStrategy.MAX_PROB: '#66c2a5',
    ExitStrategy.ENTROPY: '#fc8d62',
    ExitStrategy.EVEN_MAX_PROB: 'black',
    ExitStrategy.EVEN_MAX_PROB_RAND: 'blue'

}


def execute_for_rate(test_probs: np.ndarray,
                     val_probs: np.ndarray,
                     test_labels: np.ndarray,
                     val_labels: np.ndarray,
                     rate: float,
                     strategy: ExitStrategy,
                     rand: np.random.RandomState) -> Tuple[float, float]:
    # Get the batch result
    if strategy == ExitStrategy.MAX_PROB:
        result = max_prob_exit(test_probs=test_probs, val_probs=val_probs, rates=[1.0 - rate, rate])
    elif strategy == ExitStrategy.ENTROPY:
        result = entropy_exit(test_probs=test_probs, val_probs=val_probs, rates=[1.0 - rate, rate])
    elif strategy == ExitStrategy.RANDOM:
        result = random_exit(test_probs=test_probs, rates=[1.0 - rate, rate], rand=rand)
    elif strategy == ExitStrategy.EVEN_MAX_PROB:
        result = even_max_prob_exit(test_probs=test_probs, val_probs=val_probs, rates=[1.0 - rate, rate], val_labels=val_labels, rand=rand, use_rand=False)
    elif strategy == ExitStrategy.EVEN_MAX_PROB_RAND:
        result = even_max_prob_exit(test_probs=test_probs, val_probs=val_probs, rates=[1.0 - rate, rate], val_labels=val_labels, rand=rand, use_rand=True)
    else:
        raise ValueError('Unknown exit strategy: {}'.format(strategy))

    accuracy = compute_accuracy(result.preds, labels=test_labels)
    mutual_information = compute_mutual_info(result.output_counts, test_labels)

    return accuracy, mutual_information


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model weights')
    #parser.add_argument('--tuned-path', type=str, required=True, help='Path to the tuned model weights')
    parser.add_argument('--output-path', type=str, help='Path to save the final plot')
    args = parser.parse_args()

    # Restore the model
    model: NeuralNetwork = restore_model(path=args.model_path, model_mode=ModelMode.TEST)
    #tuned_model: NeuralNetwork = restore_model(path=args.tuned_path, model_mode=ModelMode.TEST)

    # Get the predictions from the models
    test_probs = model.test(op=OpName.PROBS)  # [B, K]
    val_probs = model.validate(op=OpName.PROBS)
    #tuned_probs = tuned_model.test(op=OpName.PROBS)

    test_labels = model.dataset.get_test_labels()
    val_labels = model.dataset.get_val_labels()

    rates = list(sorted(np.arange(0.0, 1.01, 0.1)))
    rand = np.random.RandomState(seed=591)

    # Execute all early stopping policies
    accuracy_dict: DefaultDict[ExitStrategy, List[float]] = defaultdict(list)
    info_dict: DefaultDict[ExitStrategy, List[float]] = defaultdict(list)

    strategies = [ExitStrategy.RANDOM, ExitStrategy.MAX_PROB, ExitStrategy.ENTROPY, ExitStrategy.EVEN_MAX_PROB, ExitStrategy.EVEN_MAX_PROB_RAND]

    for strategy in strategies:
        for rate in rates:
            accuracy, information = execute_for_rate(test_probs=test_probs,
                                                     val_probs=val_probs,
                                                     test_labels=test_labels,
                                                     val_labels=val_labels,
                                                     rate=rate,
                                                     strategy=strategy,
                                                     rand=rand)

            #if strategy == ExitStrategy.RANDOM:
            #    print('Rate: {:.2f}, Accuracy: {}'.format(rate, accuracy))

            # Log the results
            accuracy_dict[strategy].append(accuracy)
            info_dict[strategy].append(information)

    # Get the tuned model results
    #tuned_accuracy_list: List[float] = []
    #tuned_information_list: List[float] = []

    #tuned_stop_probs = tuned_model.test(op=OpName.STOP_PROBS)
    #for rate in rates:
    #    exit_result = random_exit_stop_probs(tuned_probs, rates=tuned_stop_probs, rand=rand)
    #    tuned_accuracy = compute_accuracy(exit_result.preds, labels=test_labels)
    #    tuned_information = compute_mutual_info(exit_result.output_counts, test_labels)

    #    #tuned_accuracy, tuned_information = execute_for_rate(tuned_probs, rate=rate, labels=test_labels, strategy=ExitStrategy.RANDOM, rand=rand)

    #    tuned_accuracy_list.append(tuned_accuracy)
    #    tuned_information_list.append(tuned_information)

    #print(tuned_accuracy_list)

    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2) = plt.subplots(figsize=(6, 4), nrows=1, ncols=2)

        for strategy in strategies:
            ax1.plot(rates, accuracy_dict[strategy], label=strategy.name.capitalize(), linewidth=3, marker='o', markersize=8, color=COLORS[strategy])
            ax2.plot(rates, info_dict[strategy], label=strategy.name.capitalize(), linewidth=3, marker='o', markersize=8, color=COLORS[strategy])

        #ax1.plot(rates, tuned_accuracy_list, label='Random (Tuned)', linewidth=3, marker='o', markersize=8, color='red')
        #ax2.plot(rates, tuned_information_list, label='Random (Tuned)', linewidth=3, marker='o', markersize=8, color='red')

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
