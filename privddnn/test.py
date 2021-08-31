import tensorflow as tf2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum, auto
from typing import List, Tuple, DefaultDict

from neural_network import restore_model, NeuralNetwork, OpName, ModelMode
from exiting.early_exit import ExitStrategy, EarlyExiter, make_policy
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info
from privddnn.utils.plotting import to_label


COLORS = {
    ExitStrategy.RANDOM: '#8da0cb',
    ExitStrategy.MAX_PROB: '#66c2a5',
    ExitStrategy.ENTROPY: '#fc8d62',
    ExitStrategy.LABEL_MAX_PROB: 'black',
    ExitStrategy.LABEL_ENTROPY: 'blue'
}


def execute_for_rate(test_probs: np.ndarray,
                     val_probs: np.ndarray,
                     test_labels: np.ndarray,
                     val_labels: np.ndarray,
                     rate: float,
                     strategy: ExitStrategy) -> Tuple[float, float]:
    # Make the exit policy
    rates = [rate, 1.0 - rate]
    policy = make_policy(strategy=strategy, rates=rates)
    policy.fit(val_probs=val_probs, val_labels=val_labels)

    # Run the policy on the test set
    result = policy.test(test_probs=test_probs)

    # Compute the result metrics
    accuracy = compute_accuracy(result.predictions, labels=test_labels)
    mutual_information = compute_mutual_info(result.output_levels, test_labels)
    observed_rate = result.observed_rates[1]  # Fraction stopping at the larger model

    return accuracy, mutual_information, observed_rate


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--output-path', type=str, help='Path to save the final plot')
    args = parser.parse_args()

    # Restore the model
    model: NeuralNetwork = restore_model(path=args.model_path, model_mode=ModelMode.TEST)

    # Get the predictions from the models
    test_probs = model.test(op=OpName.PROBS)  # [B, K]
    val_probs = model.validate(op=OpName.PROBS)

    test_labels = model.dataset.get_test_labels()
    val_labels = model.dataset.get_val_labels()

    rates = list(sorted(np.arange(0.0, 1.01, 0.1)))
    rand = np.random.RandomState(seed=591)

    # Execute all early stopping policies
    accuracy_dict: DefaultDict[ExitStrategy, List[float]] = defaultdict(list)
    info_dict: DefaultDict[ExitStrategy, List[float]] = defaultdict(list)
    rates_dict: DefaultDict[ExitStrategy, List[float]] = defaultdict(list)

    #strategies = [ExitStrategy.RANDOM, ExitStrategy.MAX_PROB, ExitStrategy.ENTROPY, ExitStrategy.EVEN_MAX_PROB]
    strategies = [ExitStrategy.RANDOM, ExitStrategy.MAX_PROB, ExitStrategy.ENTROPY, ExitStrategy.LABEL_MAX_PROB, ExitStrategy.LABEL_ENTROPY]

    for strategy in strategies:
        for rate in reversed(rates):
            accuracy, information, obs_rate = execute_for_rate(test_probs=test_probs,
                                                               val_probs=val_probs,
                                                               test_labels=test_labels,
                                                               val_labels=val_labels,
                                                               rate=rate,
                                                               strategy=strategy)

            # Log the results
            accuracy_dict[strategy].append(accuracy)
            info_dict[strategy].append(information)
            rates_dict[strategy].append(obs_rate)

    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2) = plt.subplots(figsize=(6, 4), nrows=1, ncols=2)

        for strategy in strategies:
            observed_rates = rates_dict[strategy]
            ax1.plot(observed_rates, accuracy_dict[strategy], label=to_label(strategy.name), linewidth=3, marker='o', markersize=8, color=COLORS[strategy])
            ax2.plot(observed_rates, info_dict[strategy], label=to_label(strategy.name), linewidth=3, marker='o', markersize=8, color=COLORS[strategy])

        ax1.legend()
        ax1.set_xlabel('Frac Stopping at 2nd Output')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')

        ax2.set_xlabel('Frac Stopping at 2nd Output')
        ax2.set_ylabel('Mutual Information')
        ax2.set_title('Output vs Label Mut Inf')

        plt.tight_layout()

        if args.output_path is None:
            plt.show()
        else:
            plt.savefig(args.output_path, bbox_inches='tight', transparent=True)
