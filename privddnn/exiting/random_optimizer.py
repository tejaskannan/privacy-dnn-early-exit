import numpy as np
from argparse import ArgumentParser
from typing import Tuple

from privddnn.utils.constants import SMALL_NUMBER, BIG_NUMBER
from privddnn.utils.metrics import sigmoid
from privddnn.restore import restore_classifier
from privddnn.classifier import ModelMode, BaseClassifier, OpName


MAX_ITER = 100
STEP_SIZE = 1.0
ANNEAL_RATE = 0.9
TOLERANCE = 1e-5


class RandomizedObjective:

    def __init__(self, target: float):
        self._target = target

    def __call__(self, metrics: np.ndarray, labels: np.ndarray, threshold: float, sharpness: float) -> Tuple[float, float, float]:
        probs = sigmoid(sharpness * (metrics - threshold))
        diff = np.average(probs) - self._target
        loss = np.abs(diff)

        dloss = np.sign(diff)
        dthreshold = -dloss * sharpness * np.average(probs * (1.0 - probs))
        dsharpness = dloss * np.average((metrics - threshold) * probs * (1.0 - probs))

        return loss, dthreshold, dsharpness


def fit_even_randomization(metrics: np.ndarray, labels: np.ndarray, elevate_target: float) -> float:
    rand = np.random.RandomState(935)
    objective = RandomizedObjective(target=elevate_target)

    threshold = rand.uniform(low=0.2, high=0.8)
    sharpness = rand.uniform(low=0.8, high=1.2)

    step_size = STEP_SIZE

    for _ in range(MAX_ITER):
        loss, dthreshold, dsharpness = objective(metrics=metrics,
                                                 labels=labels,
                                                 threshold=threshold,
                                                 sharpness=sharpness)
        if loss < TOLERANCE:
            break

        threshold = threshold - step_size * dthreshold
        sharpness = sharpness - step_size * dsharpness

        step_size *= ANNEAL_RATE

    return threshold, sharpness


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    args = parser.parse_args()

    # Restore the model
    model: BaseClassifier = restore_classifier(model_path=args.model_path, model_mode=ModelMode.TEST)

    # Get the predictions from the models
    val_probs = model.validate(op=OpName.PROBS)  # [B, L, K]
    val_metrics = np.max(val_probs, axis=-1)  # [B, L]
    val_metrics = val_metrics[:, 0]

    # Get the validation labels (we use this to fit any policies)
    val_labels = model.dataset.get_val_labels()  # [B]

    fit_even_randomization(metrics=val_metrics, labels=val_labels, elevate_target=0.2)
