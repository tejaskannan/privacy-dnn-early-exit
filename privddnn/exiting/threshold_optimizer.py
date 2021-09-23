import numpy as np
import scipy.optimize as optimize
from privddnn.utils.constants import SMALL_NUMBER
from privddnn.utils.metrics import create_confusion_matrix


MAX_ITERS = 150
ANNEAL_RATE = 0.99
PATIENCE = 10


class SharpenedSigmoid:

    def __init__(self, beta: float):
        self._beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-1 * self._beta * x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sigmoid = self(x)
        return self._beta * sigmoid * (1.0 - sigmoid)


class ThresholdObjective:

    def __init__(self, target: float, num_labels: int):
        self._target = target
        self._num_labels = num_labels
        self._sigmoid = SharpenedSigmoid(beta=15)

    @property
    def target(self) -> float:
        return self._target

    @property
    def num_labels(self) -> int:
        return self._num_labels

    def __call__(self, metrics: np.ndarray, preds: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> float:
        label_levels = np.zeros(shape=(self.num_labels, ))
        label_counts = np.zeros(shape=(self.num_labels, ))

        for metric, pred, label in zip(metrics, preds, labels):
            #label_levels[label] += self._sigmoid(thresholds[pred] - metric)
            label_levels[label] += int(metric < thresholds[pred])
            label_counts[label] += 1

        avg_rates = label_levels / (label_counts + SMALL_NUMBER)
        return 0.5 * float(np.sum(np.square(100.0 * (avg_rates - self.target)))), avg_rates

    def derivative(self, metrics: np.ndarray, preds: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, avg_rates: np.ndarray) -> float:
        per_label_diff = 100.0 * (avg_rates - self.target)
        #diff_sign = np.sign(per_label_diff)
        label_counts = np.bincount(labels, minlength=self.num_labels)

        gradient = np.zeros_like(thresholds)

        for metric, pred, label in zip(metrics, preds, labels):
            gradient[pred] += (per_label_diff[label] / label_counts[label]) * self._sigmoid.derivative(thresholds[pred] - metric)

        return gradient


def fit_thresholds_grad(metrics: np.ndarray, preds: np.ndarray, labels: np.ndarray, target: float, start_thresholds: np.ndarray, learning_rate: float) -> np.ndarray:
    sample_idx = np.arange(len(labels))
    rand = np.random.RandomState(seed=25893)
    num_labels = np.max(labels) + 1

    thresholds = np.copy(start_thresholds)
    objective = ThresholdObjective(target=target, num_labels=num_labels)

    loss, rates = objective(metrics=metrics, preds=preds, labels=labels, thresholds=thresholds)
    step_size = learning_rate

    best_loss = loss
    best_thresholds = np.copy(thresholds)
    best_rates = rates
    num_not_improved = 0

    print('TARGET: {}'.format(target))
    print('START LOSS: {}'.format(loss))

    for _ in range(MAX_ITERS):
        #batch_idx = rand.choice(sample_idx, size=256, replace=False)
        #batch_metrics = metrics[batch_idx]
        #batch_preds = preds[batch_idx]
        #batch_labels = labels[batch_idx]

        loss, avg_rates = objective(metrics=metrics, preds=preds, labels=labels, thresholds=thresholds)

        dthresholds = objective.derivative(metrics=metrics, preds=preds, labels=labels, thresholds=thresholds, avg_rates=avg_rates)
        thresholds -= step_size * dthresholds

        step_size *= ANNEAL_RATE

        if loss < best_loss:
            best_loss = loss
            best_thresholds = np.copy(thresholds)
            best_rates = avg_rates
            num_not_improved = 0
        else:
            num_not_improved += 1

        print('Loss: {:.6f}'.format(loss), end='\r')

        if num_not_improved > PATIENCE:
            print('\nConverged.')
            break

    print()

    return best_loss, best_thresholds, best_rates


def fit_randomization(metrics: np.ndarray, preds: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, target: float, epsilon: float) -> np.ndarray:
    # Get the observed rates, [i, j] is the avg elevation rate for label j when predicted as i
    num_labels = np.max(labels) + 1
    elevation_counts = np.zeros(shape=(num_labels, num_labels))
    total_counts = np.zeros_like(elevation_counts)

    for metric, pred, label in zip(metrics, preds, labels):
        elevation_counts[pred, label] += int(thresholds[pred] > metric)
        total_counts[pred, label] += 1

    observed_rates = elevation_counts / (total_counts + SMALL_NUMBER)

    # Get the confusion matrix, [i, j] is the number of elements classified as i that are actually j
    confusion_mat = create_confusion_matrix(predictions=preds, labels=labels)
    confusion_mat = confusion_mat / (np.sum(confusion_mat, axis=0, keepdims=True) + SMALL_NUMBER)

    # Create the constraint coefficient matrix
    A = (confusion_mat * (target - observed_rates)).T

    # Get the constraint bounds
    beta = np.diag(confusion_mat.T.dot(observed_rates))
    lower = (target - epsilon) - beta
    upper = (target + epsilon) + beta

    # Create the 'security' constraint
    constr1 = optimize.LinearConstraint(A=A, lb=lower, ub=upper)

    # Constrain the rates to the range [0, 1]
    constr2 = optimize.LinearConstraint(A=np.eye(A.shape[0]), lb=0.0, ub=1.0)

    result = optimize.minimize(fun=lambda x: np.sum(x),
                               x0=np.ones(shape=(num_labels, )),
                               method='trust-constr',
                               hess=lambda x: np.zeros_like(A),
                               constraints=[constr1, constr2],
                               tol=1e-4)

    print(result.x)
    return result.x
