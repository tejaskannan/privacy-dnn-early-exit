import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from privddnn.classifier import ModelMode, OpName
from privddnn.ensemble.adaboost import AdaBoostClassifier


def make_confusion_mat(preds: np.ndarray, num_labels: int) -> np.ndarray:
    confusion_mat = np.zeros(shape=(num_labels, num_labels), dtype=float)

    for level_preds in preds:
        level0, level1 = level_preds[0], level_preds[1]
        confusion_mat[level0, level1] += 1.0

    return confusion_mat


def plot_rates(confusion_mat: np.ndarray):
    width = 0.25
    xs = np.arange(confusion_mat.shape[0])

    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()

        level0_rates = np.sum(confusion_mat, axis=-1) / np.sum(confusion_mat)
        level1_rates = np.sum(confusion_mat, axis=0) / np.sum(confusion_mat)

        ax.bar(xs - (width / 2), height=level0_rates, width=width, label='Level 0')
        ax.bar(xs + (width / 2), height=level1_rates, width=width, label='Level 1')

        ax.legend()
        ax.set_xticks(xs)

        ax.set_title('Prediction Rates for Model Levels')
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Fraction of Instances')

        plt.savefig('../results/18-10/pen_digits_adaboost_prediction_rates.pdf', bbox_inches='tight', transparent=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model file')
    args = parser.parse_args()

    # Fit the model
    model = AdaBoostClassifier.restore(path=args.model_path, model_mode=ModelMode.TEST)

    # Get the predictions from the models
    test_probs = model.test(op=OpName.PROBS)  # [B, K]
    val_probs = model.validate(op=OpName.PROBS)

    val_confusion = make_confusion_mat(preds=np.argmax(val_probs, axis=-1), num_labels=val_probs.shape[-1])

    plot_rates(confusion_mat=val_confusion)
