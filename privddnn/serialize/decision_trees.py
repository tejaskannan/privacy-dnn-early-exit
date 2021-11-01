import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from privddnn.restore import restore_classifier
from privddnn.classifier import ModelMode
from privddnn.ensemble.adaboost import AdaBoostClassifier
from privddnn.serialize.utils import serialize_int_array, serialize_float_array


def serialize_tree(clf: DecisionTreeClassifier, name: str, precision: int, should_print: bool):
    root = clf.tree_
    lines: List[str] = []
    num_nodes = len(root.feature)

    features_name = '{}_FEATURES'.format(name)
    features = serialize_int_array(var_name=features_name,
                                   array=root.feature,
                                   dtype='int8_t')
    lines.append(features)

    thresholds_name = '{}_THRESHOLDS'.format(name)
    thresholds = serialize_float_array(var_name=thresholds_name,
                                       array=root.threshold,
                                       width=16,
                                       precision=precision,
                                       dtype='int16_t')
    lines.append(thresholds)

    left_name = '{}_CHILDREN_LEFT'.format(name)
    children_left = serialize_int_array(var_name=left_name,
                                        array=root.children_left,
                                        dtype='int8_t')
    lines.append(children_left)
                               
    right_name = '{}_CHILDREN_RIGHT'.format(name)
    children_right = serialize_int_array(var_name=right_name,
                                         array=root.children_right,
                                         dtype='int8_t')
    lines.append(children_right)

    pred_name = '{}_PREDICTIONS'.format(name)
    predictions = serialize_int_array(var_name=pred_name,
                                      array=[np.argmax(node[0]) for node in root.value],
                                      dtype='uint8_t')
    lines.append(predictions)

    tree_var = 'static struct decision_tree {} = {{ {}, {}, {}, {}, {}, {} }};'.format(name, num_nodes, thresholds_name, features_name, pred_name, left_name, right_name)
    lines.append(tree_var)

    return '\n'.join(lines)


def serialize_ensemble(ensemble: AdaBoostClassifier, precision: int) -> str:
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
        lines.append(serialize_tree(clf, name=name, precision=precision, should_print=(idx == 0)))
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


if __name__ == '__main__':
    path = '../saved_models/pen_digits/01-10-2021/decision_tree_01-10-2021-15-55-08.pkl.gz'
    ensemble = restore_classifier(path, model_mode=ModelMode.TEST)

    result = serialize_ensemble(ensemble, precision=10)

    with open('../msp430/parameters.h', 'w') as fout:
        fout.write('#include <stdint.h>\n')
        fout.write('#include "decision_tree.h"\n')

        fout.write('#ifndef PARAMETERS_H_\n')
        fout.write('#define PARAMETERS_H_\n')

        fout.write(result)

        fout.write('\n#endif')
