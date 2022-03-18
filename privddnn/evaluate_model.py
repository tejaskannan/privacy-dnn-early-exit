import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix

from privddnn.classifier import BaseClassifier, ModelMode
from privddnn.restore import restore_classifier


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    args = parser.parse_args()

    # Restore the model
    model: BaseClassifier = restore_classifier(model_path=args.model_path, model_mode=ModelMode.TEST)

    # Get the predictions from the models
    val_probs = model.validate()  # [B, L, K]
    test_probs = model.test()  # [C, L, K]

    # Get the labels from each set
    val_labels = model.dataset.get_val_labels()
    test_labels = model.dataset.get_test_labels()

    label_space = list(range(val_probs.shape[-1]))

    for output_idx in range(model.num_outputs):
        val_preds = np.argmax(val_probs[:, output_idx, :], axis=-1)
        test_preds = np.argmax(test_probs[:, output_idx, :], axis=-1)

        val_accuracy = np.average(np.isclose(val_labels, val_preds).astype(float))
        test_accuracy = np.average(np.isclose(test_labels, test_preds).astype(float))

        val_conf_mat = confusion_matrix(y_true=val_labels, y_pred=val_preds, labels=label_space, normalize='all').astype(float)
        val_conf_mat /= np.sum(val_conf_mat)
        val_conf_mat *= 100.0

        test_conf_mat = confusion_matrix(y_true=test_labels, y_pred=test_preds, labels=label_space, normalize='all').astype(float)
        test_conf_mat *= 100.0

        print('Output: {}. Val -> {:.3f}, Test -> {:.3f}'.format(output_idx, val_accuracy * 100.0, test_accuracy * 100.0))
        print('----------')
        print('Val Confusion Matrix')
        print('{}'.format(np.array_str(val_conf_mat, precision=3, suppress_small=True)))
        print('----------')
        print('Test Confusion Matrix')
        print('{}'.format(np.array_str(test_conf_mat, precision=3, suppress_small=True)))
        print('==========')
