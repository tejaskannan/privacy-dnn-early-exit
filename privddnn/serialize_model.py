import numpy as np
from argparse import ArgumentParser
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from privddnn.restore import restore_classifier
from privddnn.classifier import ModelMode, OpName
from privddnn.exiting.early_exit import make_policy, ExitStrategy
from privddnn.ensemble.adaboost import AdaBoostClassifier
from privddnn.neural_network import BranchyNetDNN
from privddnn.serialize.exit_policy import serialize_policy
from privddnn.serialize.utils import serialize_int_array, serialize_float_array, expand_vector
from privddnn.utils.file_utils import read_pickle_gz


def serialize_tree(clf: DecisionTreeClassifier, name: str, precision: int, is_msp: bool):
    root = clf.tree_
    lines: List[str] = []
    num_nodes = len(root.feature)

    features_name = '{}_FEATURES'.format(name)
    features = serialize_int_array(var_name=features_name,
                                   array=root.feature,
                                   dtype='int8_t')
    lines.append('#pragma PERSISTENT({})'.format(features_name))
    lines.append(features)

    thresholds_name = '{}_THRESHOLDS'.format(name)
    thresholds = serialize_float_array(var_name=thresholds_name,
                                       array=root.threshold,
                                       width=16,
                                       precision=precision,
                                       dtype='int16_t')
    lines.append('#pragma PERSISTENT({})'.format(thresholds_name))
    lines.append(thresholds)

    left_name = '{}_CHILDREN_LEFT'.format(name)
    children_left = serialize_int_array(var_name=left_name,
                                        array=root.children_left,
                                        dtype='int8_t')
    lines.append('#pragma PERSISTENT({})'.format(left_name))
    lines.append(children_left)

    right_name = '{}_CHILDREN_RIGHT'.format(name)
    children_right = serialize_int_array(var_name=right_name,
                                         array=root.children_right,
                                         dtype='int8_t')
    lines.append('#pragma PERSISTENT({})'.format(right_name))
    lines.append(children_right)

    pred_name = '{}_PREDICTIONS'.format(name)
    predictions = serialize_int_array(var_name=pred_name,
                                      array=[np.argmax(node[0]) for node in root.value],
                                      dtype='uint8_t')
    lines.append('#pragma PERSISTENT({})'.format(pred_name))
    lines.append(predictions)

    tree_var = 'static struct decision_tree {} = {{ {}, {}, {}, {}, {}, {} }};'.format(name, num_nodes, thresholds_name, features_name, pred_name, left_name, right_name)
    lines.append(tree_var)

    return '\n'.join(lines)


def serialize_ensemble(ensemble: AdaBoostClassifier, precision: int, is_msp: bool) -> str:
    lines: List[str] = []

    num_estimators = ensemble.num_estimators
    exit_point = ensemble.exit_size
    boost_weights = ensemble._boost_weights
    clfs = ensemble._clfs
    num_labels = np.max(ensemble.dataset.get_train_labels()) + 1

    # Collect the information for all of the classifiers
    var_names: List[str] = []

    for idx, clf in enumerate(clfs):
        name = 'TREE_{}'.format(idx)
        lines.append(serialize_tree(clf, name=name, precision=precision, is_msp=is_msp))
        var_names.append(name)

    # Create the array of decision trees
    trees_var = 'static struct decision_tree *TREES[] = {{ {} }};'.format(','.join(('&{}'.format(n) for n in var_names)))
    lines.append(trees_var)

    # Create the array of boosting weights
    boost_weights_var = serialize_float_array(var_name='BOOST_WEIGHTS',
                                              array=boost_weights,
                                              precision=precision,
                                              width=16,
                                              dtype='int16_t')
    lines.append(boost_weights_var)

    # Create the ensemble structure
    ensemble_var = 'static struct adaboost_ensemble ENSEMBLE = {{ {}, {}, {}, TREES, BOOST_WEIGHTS }};'.format(num_estimators, exit_point, num_labels)
    lines.append(ensemble_var)

    return '\n'.join(lines)


def serialize_branchynet_dnn(model_path: str, precision: int, is_msp: bool):
    model_parameters = read_pickle_gz(model_path)['weights']

    # Unpack the relevant variables for early exiting
    W0 = model_parameters['hidden1/W:0'].T
    b0 = model_parameters['hidden1/b:0'].reshape(-1)

    W1 = model_parameters['output1/W:0'].T
    b1 = model_parameters['output1/b:0'].reshape(-1)

    # Create C variables for each trainable parameter
    vec_cols = 2 if args.is_msp else 1
    lines: List[str] = []

    W0_shape = W0.shape
    W0_data = serialize_float_array(var_name='W0_DATA',
                                    array=W0.reshape(-1),
                                    width=16,
                                    precision=precision,
                                    dtype='int16_t')
    if is_msp:
        lines.append('#pragma PERSISTENT(W0_DATA)')

    lines.append(W0_data)

    W0_var = 'static struct matrix W0 = {{ W0_DATA, {}, {} }};\n'.format(W0_shape[0], W0_shape[1])
    lines.append(W0_var)

    b0_shape = b0.shape
    if is_msp:
        b0 = expand_vector(b0)

    b0_data = serialize_float_array(var_name='B0_DATA',
                                    array=b0,
                                    width=16,
                                    precision=precision,
                                    dtype='int16_t')
    if is_msp:
        lines.append('#pragma PERSISTENT(B0_DATA)')

    lines.append(b0_data)

    b0_var = 'static struct matrix B0 = {{ B0_DATA, {}, {} }};\n'.format(b0_shape[0], vec_cols)
    lines.append(b0_var)

    W1_shape = W1.shape
    W1_data = serialize_float_array(var_name='W1_DATA',
                                    array=W1.reshape(-1),
                                    width=16,
                                    precision=precision,
                                    dtype='int16_t')
    if is_msp:
        lines.append('#pragma PERSISTENT(W1_DATA)')

    lines.append(W1_data)

    W1_var = 'static struct matrix W1 = {{ W1_DATA, {}, {} }};\n'.format(W1_shape[0], W1_shape[1])
    lines.append(W1_var)

    b1_shape = b1.shape
    if is_msp:
        b1 = expand_vector(b1)

    b1_data = serialize_float_array(var_name='B1_DATA',
                                    array=b1,
                                    width=16,
                                    precision=precision,
                                    dtype='int16_t')
    if is_msp:
        lines.append('#pragma PERSISTENT(B1_DATA)')

    lines.append(b1_data)

    b1_var = 'static struct matrix B1 = {{ B1_DATA, {}, {} }};\n'.format(b1_shape[0], vec_cols)
    lines.append(b1_var)

    return '\n'.join(lines)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model parameters.')
    parser.add_argument('--exit-policy', type=str, required=True, help='Name of the exit policy to serialize.')
    parser.add_argument('--exit-rate', type=float, required=True, help='The target exit rate in [0, 1].')
    parser.add_argument('--precision', type=int, required=True, help='The precision of fixed point values.')
    parser.add_argument('--is-msp', action='store_true', help='Whether to serialize the system for the MSP430.')
    args = parser.parse_args()

    assert args.exit_rate >= 0.0 and args.exit_rate <= 1.0, 'The exit rate must be in [0, 1].'

    clf = restore_classifier(args.model_path, model_mode=ModelMode.TEST)

    strategy = ExitStrategy[args.exit_policy.upper()]
    rates = [args.exit_rate, 1.0 - args.exit_rate]
    policy = make_policy(strategy=strategy, rates=rates, model_path=args.model_path)

    val_probs = clf.validate(op=OpName.PROBS)
    val_labels = clf.dataset.get_val_labels()
    policy.fit(val_probs, val_labels)

    num_labels = clf.dataset.num_labels
    num_input_features = clf.dataset.num_features

    if isinstance(clf, AdaBoostClassifier):
        serialized_model = serialize_ensemble(clf, precision=args.precision, is_msp=args.is_msp)
        model_type = 'IS_ADABOOST'
    elif isinstance(clf, BranchyNetDNN):
        serialized_model = serialize_branchynet_dnn(args.model_path, precision=args.precision, is_msp=args.is_msp)
        model_type = 'IS_DNN'
    else:
        raise ValueError('Serialization does not support the classifier {}'.format(clf.name))

    save_dir = 'msp430' if args.is_msp else 'c_implementation'

    with open('{}/parameters.h'.format(save_dir), 'w') as fout:
        fout.write('#include <stdint.h>\n')
        fout.write('#include "decision_tree.h"\n')

        fout.write('#ifndef PARAMETERS_H_\n')
        fout.write('#define PARAMETERS_H_\n')

        fout.write(serialize_policy(policy, precision=args.precision))
        fout.write('\n')
        fout.write('#define {}\n'.format(model_type))
        fout.write('#define PRECISION {}\n'.format(args.precision))
        fout.write('#define NUM_LABELS {}\n'.format(num_labels))
        fout.write('#define NUM_INPUT_FEATURES {}\n\n'.format(num_input_features))

        fout.write(serialized_model)

        fout.write('\n#endif\n')
