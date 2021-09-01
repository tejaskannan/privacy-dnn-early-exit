import numpy as np
import os.path
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum, auto
from typing import List, Tuple, DefaultDict

from neural_network import restore_model, NeuralNetwork, OpName, ModelMode
from exiting.early_exit import ExitStrategy, EarlyExiter, make_policy
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info
from privddnn.utils.plotting import to_label
from privddnn.utils.file_utils import save_json


COLORS = {
    ExitStrategy.RANDOM: '#756bb1',
    ExitStrategy.MAX_PROB: '#3182bd',
    ExitStrategy.ENTROPY: '#31a354',
    ExitStrategy.LABEL_MAX_PROB: '#9ecae1',
    ExitStrategy.LABEL_ENTROPY: '#a1d99b'
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

    # Fit the attack model
    val_result = policy.test(test_probs=val_probs)
    policy.fit_attack_model(val_outputs=val_result.output_levels,
                            val_labels=val_labels,
                            window_size=10,
                            num_samples=1000)

    attack_accuracy = policy.test_attack_model(test_outputs=result.output_levels,
                                               test_labels=test_labels,
                                               window_size=10,
                                               num_samples=1000)

    return accuracy, mutual_information, observed_rate, attack_accuracy


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
    accuracy_dict: DefaultDict[str, List[float]] = defaultdict(list)
    info_dict: DefaultDict[str, List[float]] = defaultdict(list)
    rates_dict: DefaultDict[str, List[float]] = defaultdict(list)
    attack_dict: DefaultDict[str, List[float]] = defaultdict(list)

    #strategies = [ExitStrategy.RANDOM, ExitStrategy.MAX_PROB, ExitStrategy.ENTROPY, ExitStrategy.LABEL_MAX_PROB, ExitStrategy.LABEL_ENTROPY]
    strategies = [ExitStrategy.LABEL_MAX_PROB, ExitStrategy.RANDOM]
    rates = [0.4, 0.5]

    for strategy in strategies:
        for rate in reversed(rates):
            accuracy, information, obs_rate, attack_accuracy = execute_for_rate(test_probs=test_probs,
                                                                                val_probs=val_probs,
                                                                                test_labels=test_labels,
                                                                                val_labels=val_labels,
                                                                                rate=rate,
                                                                                strategy=strategy)

            # Log the results
            strategy_name = strategy.name.lower()
            accuracy_dict[strategy_name].append(accuracy)
            info_dict[strategy_name].append(information)
            rates_dict[strategy_name].append(obs_rate)
            attack_dict[strategy_name].append(attack_accuracy)

    # Save the results into the test log
    file_name = os.path.basename(args.model_path).split('.')[0]
    test_log_name = '{}_test-log.json'.format(file_name)
    test_log_path = os.path.join(os.path.dirname(args.model_path), test_log_name)

    test_log = {
        'accuracy': accuracy_dict,
        'mutual_information': info_dict,
        'observed_rates': rates_dict,
        'attack_accuracy': attack_dict
    }

    save_json(test_log, test_log_path)
