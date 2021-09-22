import os.path
import numpy as np
from argparse import ArgumentParser
from typing import List, Dict

from neural_network import restore_model, NeuralNetwork, OpName, ModelMode
from exiting.early_exit import OptimizedMaxProb
from privddnn.utils.metrics import create_confusion_matrix
from privddnn.utils.file_utils import save_json


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    args = parser.parse_args()

    # Restore the model
    model: NeuralNetwork = restore_model(path=args.model_path, model_mode=ModelMode.TEST)

    val_probs = model.validate(op=OpName.PROBS)
    val_labels = model.dataset.get_val_labels()

    # Make the policy
    #rates = list([(x / 10.0) for x in range(11)])
    rates = [0.3]

    thresholds_dict: Dict[float, List[List[float]]] = dict()
    rates_dict: Dict[float, float] = dict()

    for rate in rates:
        print('===== Rate {:.2f} ====='.format(rate))

        exit_policy = OptimizedMaxProb(rates=[rate, 1.0 - rate], path=None)
        exit_policy.fit(val_probs=val_probs, val_labels=val_labels)

        thresholds_dict[round(rate, 2)] = exit_policy.thresholds.tolist()
        rates_dict[round(rate, 2)] = exit_policy._rand_rate
        print()

    sample_probs = create_confusion_matrix(predictions=np.argmax(val_probs[:, 0, :], axis=-1),
                                           labels=val_labels)

    result = {
        'thresholds': thresholds_dict,
        'rates': rates_dict,
        'prob_std': float(np.std(np.max(val_probs[:, 0, :], axis=-1))),
        'sample_probs': sample_probs.astype(float).tolist()
    }

    # Get the model file name
    folder = os.path.dirname(args.model_path)
    file_name = os.path.basename(args.model_path)
    model_name = file_name.split('.')[0]
    out_file_name = '{}_max-prob-thresholds.json'.format(model_name)

    # Save the result
    save_json(result, os.path.join(folder, out_file_name))
