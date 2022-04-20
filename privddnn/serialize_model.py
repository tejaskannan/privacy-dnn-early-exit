import numpy as np
import h5py
from argparse import ArgumentParser
from typing import List

from privddnn.restore import restore_classifier
from privddnn.classifier import ModelMode
from privddnn.exiting.early_exit import make_policy, ExitStrategy
from privddnn.serialize.exit_policy import serialize_policy
from privddnn.serialize.utils import serialize_int_array, serialize_float_array, expand_vector
from privddnn.utils.constants import SMALL_NUMBER
from privddnn.utils.file_utils import read_pickle_gz


def serialize_dense_layer(weight_mat: np.ndarray, bias: np.ndarray, is_msp: bool, name: str, precision: int) -> List[str]:
    result: List[str] = []

    weight_mat = weight_mat.T  # We operate on the transpose in the C implementation (multiply from the left)
    bias = bias.reshape(-1)

    weight_mat_shape = weight_mat.shape
    weight_data = serialize_float_array(var_name='{}_W_DATA'.format(name),
                                        array=weight_mat.reshape(-1),
                                        width=16,
                                        precision=precision,
                                        dtype='int16_t')

    if is_msp:
        result.append('#pragma PERSISTENT({}_W_DATA)'.format(name))

    result.append(weight_data)

    weight_var = 'static struct matrix {}_W = {{ {}_W_DATA, {}, {} }};'.format(name, name, weight_mat_shape[0], weight_mat_shape[1])
    result.append(weight_var)

    vec_cols = 2 if is_msp else 1

    if is_msp:
        bias = expand_vector(bias)

    bias_data = serialize_float_array(var_name='{}_B_DATA'.format(name),
                                      array=bias,
                                      width=16,
                                      precision=precision,
                                      dtype='int16_t')

    result.append(bias_data)

    bias_var = 'static struct matrix {}_B = {{ {}_B_DATA, {}, {} }};'.format(name, name, bias.shape[0], vec_cols)
    result.append(bias_var)

    return result


def serialize_branchynet_dnn(model_path: str, precision: int, is_msp: bool):
    with h5py.File(model_path, 'r') as fin:
        model_parameters = fin['model_weights']

        # Unpack the relevant variables for early exiting
        weight_matrices: Dict[str, np.ndarray] = dict()
        biases: Dict[str, np.ndarray] = dict()

        for layer_name, layer_params in model_parameters.items():
            if layer_name.startswith('dense') or layer_name.startswith('output'):
                weight_mat = layer_params[layer_name]['kernel:0'][:]
                bias = layer_params[layer_name]['bias:0'][:]

                weight_matrices[layer_name] = weight_mat
                biases[layer_name] = bias

    lines: List[str] = []

    for layer_name in weight_matrices.keys():
        serialized_layer = serialize_dense_layer(weight_mat=weight_matrices[layer_name],
                                                 bias=biases[layer_name],
                                                 is_msp=is_msp,
                                                 name=layer_name.upper(),
                                                 precision=precision)
        lines.extend(serialized_layer)
        lines.append('\n')

    return '\n'.join(lines)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model parameters.')
    parser.add_argument('--exit-policy', type=str, required=True, help='Name of the exit policy to serialize.')
    parser.add_argument('--exit-rates', type=float, required=True, nargs='+', help='The target exit rates.')
    parser.add_argument('--precision', type=int, required=True, help='The precision of fixed point values.')
    parser.add_argument('--is-msp', action='store_true', help='Whether to serialize the system for the MSP430.')
    args = parser.parse_args()

    assert abs(sum(args.exit_rates) - 1.0) < SMALL_NUMBER, 'The exit rates must sum to 1'
    assert 'branchynet-dnn' in args.model_path, 'Must give a branchynet DNN'

    clf = restore_classifier(args.model_path, model_mode=ModelMode.TEST)

    strategy = ExitStrategy[args.exit_policy.upper()]
    policy = make_policy(strategy=strategy,
                         rates=args.exit_rates,
                         model_path=args.model_path)

    val_probs = clf.validate()
    val_labels = clf.dataset.get_val_labels()
    policy.fit(val_probs, val_labels)

    num_labels = clf.dataset.num_labels
    num_input_features = clf.dataset.num_features
    num_outputs = clf.num_outputs

    serialized_model = serialize_branchynet_dnn(args.model_path, precision=args.precision, is_msp=args.is_msp)

    save_dir = 'msp430' if args.is_msp else 'c_implementation'

    with open('{}/parameters.h'.format(save_dir), 'w') as fout:
        fout.write('#include <stdint.h>\n')
        fout.write('#include "matrix.h"\n')

        fout.write('#ifndef PARAMETERS_H_\n')
        fout.write('#define PARAMETERS_H_\n')

        fout.write(serialize_policy(policy, precision=args.precision))
        fout.write('\n')
        fout.write('#define PRECISION {}\n'.format(args.precision))
        fout.write('#define NUM_LABELS {}\n'.format(num_labels))
        fout.write('#define NUM_INPUT_FEATURES {}\n'.format(num_input_features))
        fout.write('#define NUM_OUTPUTS {}\n\n'.format(num_outputs))

        fout.write(serialized_model)

        fout.write('\n#endif\n')
