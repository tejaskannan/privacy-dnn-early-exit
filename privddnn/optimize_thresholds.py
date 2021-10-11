import os.path
import numpy as np
from argparse import ArgumentParser
from typing import List, Dict

from neural_network import restore_model, NeuralNetwork
from exiting.early_exit import ExitStrategy, make_policy
from privddnn.classifier import OpName, ModelMode
from privddnn.ensemble.adaboost import AdaBoostClassifier
from privddnn.utils.metrics import create_confusion_matrix
from privddnn.utils.file_utils import save_json
from privddnn.utils.constants import SMALL_NUMBER


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--metric-name', type=str, required=True, choices=['entropy', 'max-prob'])
    args = parser.parse_args()

    # Restore the model
    #model: NeuralNetwork = restore_model(path=args.model_path, model_mode=ModelMode.TEST)
    model = AdaBoostClassifier.restore(path=args.model_path, model_mode=ModelMode.TEST)

    val_probs = model.validate(op=OpName.PROBS)
    val_labels = model.dataset.get_val_labels()

    # Set the target rates
    rates = list([(x / 10.0) for x in range(11)])
    rates = [0.8]

    # Get the policy class
    policy_type = ExitStrategy.HYBRID_MAX_PROB if args.metric_name == 'max-prob' else ExitStrategy.HYBRID_ENTROPY

    thresholds_dict: Dict[float, List[List[float]]] = dict()
    weights_dict: Dict[float, List[List[float]]] = dict()
    rates_dict: Dict[float, float] = dict()

    for rate in rates:
        print('===== Rate {:.2f} ====='.format(rate))

        exit_policy = make_policy(strategy=policy_type, rates=[rate, 1.0 - rate], model_path=None)
        exit_policy.fit(val_probs=val_probs, val_labels=val_labels)

        thresholds_dict[round(rate, 2)] = exit_policy.thresholds.tolist()
        weights_dict[round(rate, 2)] = exit_policy._weights.tolist()
        rates_dict[round(rate, 2)] = float(exit_policy._rand_rate)
        print()

    result = {
        'thresholds': thresholds_dict,
        'weights': weights_dict,
        'rand_rate': rates_dict
    }

    # Get the model file name
    folder = os.path.dirname(args.model_path)
    file_name = os.path.basename(args.model_path)
    model_name = file_name.split('.')[0]
    out_file_name = '{}_{}-thresholds.json'.format(model_name, args.metric_name)

    # Save the result
    save_json(result, os.path.join(folder, out_file_name))
