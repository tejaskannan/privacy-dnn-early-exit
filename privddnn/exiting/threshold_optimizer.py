import numpy as np
import scipy.optimize as optimize
from typing import Tuple

from privddnn.utils.constants import SMALL_NUMBER
from privddnn.utils.metrics import create_confusion_matrix


MAX_ITERS = 250
ANNEAL_RATE = 0.9
PATIENCE = 15


class SharpenedSigmoid:

    def __init__(self, beta: float):
        self._beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-1 * self._beta * x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sigmoid = self(x)
        return self._beta * sigmoid * (1.0 - sigmoid)


class ThresholdFunction:

    def __call__(self, x: float) -> float:
        return 1.0 if x > 0 else 0.0

    def derivative(self, x: float) -> float:
        if (x >= -1) and (x < -0.4):
            return 1
        elif (x > 0.4) and (x <= 1.0):
            return 1
        elif (x >= -0.4) and (x <= 0.4):
            return 2 - 4 * abs(x)
        
        return 0

class ThresholdObjective:

    def __init__(self, target: float, num_labels: int):
        self._target = target
        self._num_labels = num_labels
        self._sigmoid = SharpenedSigmoid(beta=1)
        self._threshold = ThresholdFunction()

    @property
    def target(self) -> float:
        return self._target

    @property
    def num_labels(self) -> int:
        return self._num_labels

    def __call__(self, probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> float:
        label_levels = np.zeros(shape=(self.num_labels, ))
        label_counts = np.zeros(shape=(self.num_labels, ))

        for pred_probs, label in zip(probs, labels):
            pred = np.argmax(pred_probs)
            metric = pred_probs[pred]

            #label_levels[label] += self._sigmoid(thresholds[pred] - stop_prob)
            #label_levels[label] += int(stop_prob < thresholds[pred])
            label_levels[label] += self._threshold(thresholds[pred] - metric)
            label_counts[label] += 1

        avg_rates = label_levels / (label_counts + SMALL_NUMBER)
        return float(np.sum(np.abs(avg_rates - self.target))), avg_rates

        #return 0.5 * float(np.sum(np.square(avg_rates - self.target))), avg_rates

    def derivative(self, probs: np.ndarray, labels: np.ndarray, avg_rates: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        #per_label_diff = (avg_rates - self.target)
        per_label_diff = 2.0 * (avg_rates > self.target).astype(float) - 1.0

        threshold_gradient = np.zeros_like(thresholds)
        num_samples = probs.shape[0]
        #weight_gradient = np.zeros_like(weights)

        label_counts = np.bincount(labels, minlength=self._num_labels)

        for pred_probs, label in zip(probs, labels):
            pred = np.argmax(pred_probs)
            metric = pred_probs[pred]

            #dL_dt = (per_label_diff[label] / label_counts[label]) * self._sigmoid.derivative(thresholds[pred] - stop_prob)
            dL_dt = (per_label_diff[label] / label_counts[label]) * self._threshold.derivative(thresholds[pred] - metric)
            #dL_ds = -1 * dL_dt
            #dL_dw = dL_ds * pred_probs[pred] * self._sigmoid.derivative(transform) * pred_probs

            threshold_gradient[pred] += dL_dt
            #weight_gradient[pred] += dL_dw

        #return threshold_gradient, weight_gradient
        return threshold_gradient


def fit_thresholds_grad(probs: np.ndarray, labels: np.ndarray, target: float, start_thresholds: np.ndarray, start_weights: np.ndarray, learning_rate: float) -> np.ndarray:
    sample_idx = np.arange(len(labels))
    rand = np.random.RandomState(seed=25893)
    num_labels = np.max(labels) + 1

    thresholds = np.copy(start_thresholds)
    #weights = np.copy(start_weights)
    #weights = rand.uniform(low=-0.7, high=0.7, size=(num_labels, num_labels))
    objective = ThresholdObjective(target=target, num_labels=num_labels)

    preds = np.argmax(probs, axis=-1)
    confusion_mat = create_confusion_matrix(predictions=preds, labels=labels) + 1
    sample_probs = confusion_mat / np.sum(confusion_mat, axis=-1)

    loss, rates = objective(probs=probs, labels=labels, thresholds=thresholds)
    step_size = learning_rate

    print('Starting Rates: {}'.format(rates))
    print('Loss: {}'.format(loss))

    best_loss = loss
    best_thresholds = np.copy(thresholds)
    #best_weights = np.copy(weights)
    best_rates = rates
    num_not_improved = 0

    expected_threshold_grad = np.zeros_like(thresholds)
    #expected_weight_grad = np.zeros_like(weights)
    gamma = 0.9
    anneal_patience = 2

    for _ in range(MAX_ITERS):
        #batch_idx = rand.choice(sample_idx, size=256, replace=False)
        #batch_metrics = metrics[batch_idx]
        #batch_preds = preds[batch_idx]
        #batch_labels = labels[batch_idx]

        loss, avg_rates = objective(probs=probs, labels=labels, thresholds=thresholds)

        # Compute the gradients
        dthresholds = objective.derivative(probs=probs, labels=labels, avg_rates=avg_rates, thresholds=thresholds)

        #approx_grad = np.zeros_like(thresholds)
        #epsilon = 1e-5

        #for i in range(num_labels):
        #    thresholds[i] += epsilon
        #    upper_loss = objective(probs=probs, labels=labels, sample_probs=sample_probs, thresholds=thresholds)

        #    thresholds[i] -= 2 * epsilon
        #    lower_loss = objective(probs=probs, labels=labels, sample_probs=sample_probs, thresholds=thresholds)

        #    thresholds[i] += epsilon
        #    approx_grad[i] = (upper_loss - lower_loss) / (2 * epsilon)

        #print('True: {}'.format(dthresholds))
        #print('Approx: {}'.format(approx_grad))

        # Update the thresholds via RMSProp
        expected_threshold_grad = gamma * expected_threshold_grad + (1.0 - gamma) * np.square(dthresholds)
        scaled_step_size = step_size / np.sqrt(expected_threshold_grad + SMALL_NUMBER)
        thresholds -= scaled_step_size * dthresholds

        #expected_weight_grad = gamma * expected_weight_grad + (1.0 - gamma) * np.square(dweights)
        #scaled_step_size = step_size / np.sqrt(expected_weight_grad + SMALL_NUMBER)
        #weights -= scaled_step_size * dweights

        #step_size *= ANNEAL_RATE

        if loss < best_loss:
            best_loss = loss
            best_thresholds = np.copy(thresholds)
            #best_weights = np.copy(weights)
            best_rates = avg_rates
            num_not_improved = 0
        else:
            num_not_improved += 1

        print('Loss: {:.6f}, Best Loss: {:.6f}'.format(loss, best_loss), end='\r')

        if (num_not_improved + 1) % anneal_patience == 0:
            step_size *= ANNEAL_RATE

        if num_not_improved > PATIENCE:
            print('\nConverged.')
            break

    print()
    print('Final Rates: {}'.format(best_rates))

    return best_loss, best_thresholds, best_rates


def fit_randomization(probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, target: float, epsilon: float) -> np.ndarray:
    # Get the observed rates, [i, j] is the avg elevation rate for label j when predicted as i
    num_labels = np.max(labels) + 1
    elevation_counts = np.zeros(shape=(num_labels, num_labels))
    total_counts = np.zeros_like(elevation_counts)

    for pred_probs, label in zip(probs, labels):
        pred = np.argmax(pred_probs)
        metric = pred_probs[pred]

        elevation_counts[pred, label] += int(thresholds[pred] > metric)
        total_counts[pred, label] += 1

    observed_rates = elevation_counts / (total_counts + SMALL_NUMBER)

    # Get the confusion matrix, [i, j] is the number of elements classified as i that are actually j
    # Each column sums to 1
    preds = np.argmax(probs, axis=-1)
    confusion_mat = create_confusion_matrix(predictions=preds, labels=labels)
    confusion_mat = confusion_mat / (np.sum(confusion_mat, axis=0, keepdims=True) + SMALL_NUMBER)

    # Create the constraint coefficient matrix
    A = (confusion_mat * (target - observed_rates)).T

    # Get the constraint bounds
    #beta = np.diag(confusion_mat.T.dot(observed_rates))
    beta = np.sum(confusion_mat * observed_rates, axis=0)
    lower = (target - epsilon) - beta
    upper = (target + epsilon) - beta

    # Create the 'security' distribution constraint
    constr1 = optimize.LinearConstraint(A=A, lb=lower, ub=upper)

    # Constrain the rates to the range [0, 1]
    constr2 = optimize.LinearConstraint(A=np.eye(A.shape[0]), lb=0.0, ub=1.0)

    result = optimize.minimize(fun=lambda x: np.sum(x),
                               x0=np.ones(shape=(num_labels, )),
                               method='trust-constr',
                               hess=lambda x: np.zeros_like(A),
                               constraints=[constr1, constr2],
                               tol=SMALL_NUMBER)

    print('Adjusted Rates: {}'.format(A.dot(result.x) + beta))
    print('Rand Rates: {}'.format(result.x))

    return result.x
