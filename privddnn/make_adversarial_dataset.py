import os.path
import numpy as np
import h5py
from argparse import ArgumentParser
from typing import List, Tuple

from privddnn.classifier import BaseClassifier, ModelMode, OpName
from privddnn.dataset import Dataset
from privddnn.exiting import ExitStrategy, EarlyExiter, make_policy, EarlyExitResult
from privddnn.utils.file_utils import make_dir
from privddnn.restore import restore_classifier


def make_adversarial_dataset(model: BaseClassifier, exit_policy: EarlyExiter, fold: str) -> Tuple[np.ndarray, np.ndarray]:
    assert fold in ('train', 'val', 'test'), 'Fold must be in (`train`, `val`, `test`)'

    # Unpack the dataset and number of labels
    dataset = model.dataset
    num_labels = dataset.num_labels

    if fold == 'train':
        inputs = dataset.get_train_inputs()
        probs = model.compute_probs(inputs=inputs)
        labels = dataset.get_train_labels()
    elif fold == 'val':
        inputs = dataset.get_val_inputs()
        probs = model.validate()
        labels = dataset.get_val_labels()
    else:
        inputs = dataset.get_test_inputs()
        probs = model.test()
        labels = dataset.get_test_labels()

    inputs_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for target_label in range(num_labels):
        exit_policy.reset()

        data_window = np.empty(shape=(num_labels, ) + inputs.shape[1:])
        label_window = np.empty(shape=(num_labels, ))

        has_exited_early = False
        has_used_full = False

        for sample_input, sample_probs, label in zip(inputs, probs, labels):
            # Run the exit policy on this sample
            level = exit_policy.select_output(probs=sample_probs)
            pred = exit_policy.get_prediction(probs=sample_probs, level=level)

            # Only look at samples with the current target prediction
            if pred != target_label:
                continue

            if (not has_exited_early) and (level == 0):
                for idx in range(num_labels):
                    if idx != pred:
                        data_window[idx] = sample_input
                        label_window[idx] = label

                has_exited_early = True

            if (not has_used_full) and (level == 1):
                data_window[pred] = sample_input
                label_window[pred] = label

                has_used_full = True

            if has_exited_early and has_used_full:
                break

        # Add the current window to the entire dataset
        inputs_list.append(data_window)
        labels_list.append(label_window)

    return np.vstack(inputs_list), np.vstack(labels_list).reshape(-1)


def write_fold(inputs: np.ndarray, labels: np.ndarray, path: str):
    with h5py.File(path, 'w') as fout:
        input_ds = fout.create_dataset('inputs', shape=inputs.shape, dtype='f')
        input_ds.write_direct(inputs)

        labels_ds = fout.create_dataset('labels', shape=labels.shape, dtype='i')
        labels_ds.write_direct(labels)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    args = parser.parse_args()

    # Restore the model
    model: BaseClassifier = restore_classifier(model_path=args.model_path, model_mode=ModelMode.TEST)

    # Make the policy
    assert model.num_outputs == 2, 'This script only works with 2-output models'
    exit_rates = [(1.0 / model.num_outputs) for _ in range(model.num_outputs)]
    exit_policy: EarlyExiter = make_policy(strategy=ExitStrategy[args.policy.upper()],
                                           rates=exit_rates,
                                           model_path=args.model_path)

    exit_policy.fit(val_probs=model.validate(),
                    val_labels=model.dataset.get_val_labels())

    # Make the output directory
    output_dir = os.path.join('data', '{}_{}'.format(model.dataset.dataset_name, args.policy))
    make_dir(output_dir)

    train_inputs, train_labels = make_adversarial_dataset(model=model, exit_policy=exit_policy, fold='train')
    val_inputs, val_labels = make_adversarial_dataset(model=model, exit_policy=exit_policy, fold='val')
    test_inputs, test_labels = make_adversarial_dataset(model=model, exit_policy=exit_policy, fold='test')

    write_fold(inputs=train_inputs, labels=train_labels, path=os.path.join(output_dir, 'train.h5'))
    write_fold(inputs=val_inputs, labels=val_labels, path=os.path.join(output_dir, 'val.h5'))
    write_fold(inputs=val_inputs, labels=test_labels, path=os.path.join(output_dir, 'test.h5'))



