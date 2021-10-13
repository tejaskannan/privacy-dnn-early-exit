import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from typing import List

from privddnn.ensemble.adaboost import AdaBoostClassifier
from privddnn.classifier import OpName, ModelMode


def range_masked_accuracy(probs: np.ndarray, labels: np.ndarray, lower: float, upper: float) -> float:
    total_count = 0
    correct_count = 0

    mask_list: List[float] = []
    for pred_probs, label in zip(probs, labels):
        for idx, p in enumerate(pred_probs):
            if p > lower and p < upper:
                total_count += 1
                correct_count += int(idx == label)

    return float(correct_count / total_count)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    args = parser.parse_args()

    model = AdaBoostClassifier.restore(path=args.model_path, model_mode=ModelMode.TEST)

    val_probs = model.validate(op=OpName.PROBS)[:, 0, :]
    val_labels = model.dataset.get_val_labels()

    #calibrate_model = LogisticRegression(C=1.0)
    #calibrate_model.fit(val_probs, val_labels)
    #val_probs = calibrate_model.predict_proba(val_probs)

    xs: List[float] = []
    ys: List[float] = []

    for edge in range(10):
        lower = edge / 10.0
        upper = (edge + 1) / 10.0

        ys.append(range_masked_accuracy(probs=val_probs, labels=val_labels, lower=lower, upper=upper))
        xs.append((upper + lower) / 2.0)

    fig, ax = plt.subplots()
    ax.scatter(xs, ys, marker='o')
    plt.show()
