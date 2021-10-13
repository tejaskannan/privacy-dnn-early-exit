import numpy as np
import os.path
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum, auto
from typing import List, Tuple, DefaultDict, Dict

from neural_network import restore_model, NeuralNetwork, OpName, ModelMode
from exiting.early_exit import ExitStrategy, EarlyExiter, make_policy, EarlyExitResult
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info, compute_target_exit_rates
from privddnn.utils.plotting import to_label
from privddnn.utils.file_utils import save_json_gz


COLORS = {
    ExitStrategy.RANDOM: '#756bb1',
    ExitStrategy.MAX_PROB: '#3182bd',
    ExitStrategy.ENTROPY: '#31a354',
    ExitStrategy.LABEL_MAX_PROB: '#9ecae1',
    ExitStrategy.LABEL_ENTROPY: '#a1d99b',
}


def execute_for_rate(test_probs: np.ndarray,
                     val_probs: np.ndarray,
                     test_labels: np.ndarray,
                     val_labels: np.ndarray,
                     rate: float,
                     model_path: str,
                     strategy: ExitStrategy,
                     num_trials: int) -> Dict[str, Dict[str, List[float]]]:
    # Make the exit policy
    rates = [rate, 1.0 - rate]
    policy = make_policy(strategy=strategy, rates=rates, model_path=model_path)
    policy.fit(val_probs=val_probs, val_labels=val_labels)

    target_exit_rates = compute_target_exit_rates(probs=val_probs, rates=rates)

    # Run the policy on the validation and test sets
    val_result = policy.test(test_probs=val_probs, target_exit_rates=target_exit_rates)
    test_result = policy.test(test_probs=test_probs, target_exit_rates=target_exit_rates)

    result = dict(val=list(), test=list())

    for _ in range(num_trials):
        result['val'].append(dict(preds=val_result.predictions.tolist(), output_levels=val_result.output_levels.tolist()))
        result['test'].append(dict(preds=test_result.predictions.tolist(), output_levels=test_result.output_levels.tolist()))

    return result

    ## Compute the result metrics
    #accuracy = compute_accuracy(result.predictions, labels=test_labels)
    #mutual_information = compute_mutual_info(result.output_levels, test_labels)
    #observed_rate = result.observed_rates[1]  # Fraction stopping at the larger model

    ## Fit the attack model
    #val_result = policy.test(test_probs=val_probs)
    #policy.fit_attack_model(val_outputs=val_result.output_levels,
    #                        val_labels=val_labels,
    #                        window_size=10,
    #                        num_samples=1000)

    #attack_accuracy = policy.test_attack_model(test_outputs=result.output_levels,
    #                                           test_labels=test_labels,
    #                                           window_size=10,
    #                                           num_samples=1000)

    #return accuracy, mutual_information, observed_rate, attack_accuracy


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--trials', type=int, required=True, help='The number of trials per policy.')
    args = parser.parse_args()

    assert args.trials >= 1, 'Must provide a positive number of trials'

    # Restore the model
    model: NeuralNetwork = restore_model(path=args.model_path, model_mode=ModelMode.TEST)

    # Get the predictions from the models
    test_probs = model.test(op=OpName.PROBS)  # [B, K]
    val_probs = model.validate(op=OpName.PROBS)

    test_labels = model.dataset.get_test_labels()
    val_labels = model.dataset.get_val_labels()

    rates = list(sorted(np.arange(0.0, 1.01, 0.05)))
    rand = np.random.RandomState(seed=591)

    # Execute all early stopping policies
    results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = dict(val=dict(), test=dict())

    #strategies = [ExitStrategy.MAX_PROB, ExitStrategy.ENTROPY, ExitStrategy.LABEL_MAX_PROB, ExitStrategy.LABEL_ENTROPY, ExitStrategy.HYBRID_MAX_PROB, ExitStrategy.HYBRID_ENTROPY, ExitStrategy.RANDOM]
    strategies = [ExitStrategy.ENTROPY, ExitStrategy.MAX_PROB, ExitStrategy.RANDOM]

    for strategy in strategies:
        strategy_name = strategy.name.lower()
        val_results: Dict[str, Dict[str, List[float]]] = dict()
        test_results: Dict[str, Dict[str, List[float]]] = dict()

        for rate in reversed(rates):
            rate_result = execute_for_rate(test_probs=test_probs,
                                           val_probs=val_probs,
                                           test_labels=test_labels,
                                           val_labels=val_labels,
                                           rate=rate,
                                           model_path=args.model_path,
                                           strategy=strategy,
                                           num_trials=args.trials)

            # Log the results
            rate_key = str(round(rate, 2))
            val_results[rate_key] = rate_result['val']
            test_results[rate_key] = rate_result['test']

        results['val'][strategy_name] = val_results
        results['test'][strategy_name] = test_results

    # Save the results into the test log
    file_name = os.path.basename(args.model_path).split('.')[0]
    test_log_name = '{}_test-log.json.gz'.format(file_name)
    test_log_path = os.path.join(os.path.dirname(args.model_path), test_log_name)

    save_json_gz(results, test_log_path)
