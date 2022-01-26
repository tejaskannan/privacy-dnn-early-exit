import numpy as np
import scipy.optimize as optimize
from typing import Tuple

from privddnn.utils.constants import SMALL_NUMBER, BIG_NUMBER
from privddnn.utils.metrics import create_confusion_matrix, compute_entropy_metric, compute_max_prob_metric


MAX_ITERS = 250
ANNEAL_RATE = 0.9
PATIENCE = 15
BETA = 5


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


class SharpnessObjective:

    def __init__(self, metric_name: str, epsilon: float, target: float, num_labels: int):
        self._metric_name = metric_name
        self._epsilon = epsilon
        self._target = target
        self._num_labels = num_labels
        self._sigmoid = SharpenedSigmoid(beta=1)

    @property
    def metric_name(self) -> str:
        return self._metric_name

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def target(self) -> float:
        return self._target

    @property
    def num_labels(self) -> int:
        return self._num_labels

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        if self.metric_name == 'entropy':
            return compute_entropy_metric(probs)
        elif self.metric_name == 'max-prob':
            return compute_max_prob_metric(probs)
        else:
            raise ValueError('Unknown metric: {}'.format(self.metric_name))

    def __call__(self, weights: np.ndarray) -> float:
        return 0.5 * np.sum(np.square(weights))

    def derivative(self, weights: np.ndarray) -> np.ndarray:
        return weights

    def is_satisfied(self, probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, weights: np.ndarray) -> Tuple[bool, np.ndarray]:
        # Compute the expected L1 distance between stopping rates
        label_levels = np.zeros(shape=(self.num_labels, ))
        label_counts = np.zeros(shape=(self.num_labels, ))

        preds = np.argmax(probs, axis=-1)
        metrics = self.compute_metric(probs=probs)

        for pred, metric, label in zip(preds, metrics, labels):
            w = np.square(weights[pred])
            elevation_prob = self._sigmoid(w * (thresholds[pred] - metric))

            label_levels[label] += elevation_prob
            label_counts[label] += 1

        avg_rates = label_levels / (label_counts + SMALL_NUMBER)

        # Compute the loss as a weighted average based on label frequency
        total_count = len(preds)
        label_weights = label_counts / total_count

        diff = float(np.sum(label_weights * np.abs(avg_rates - self.target)))

        #print('Avg Rates: {}'.format(avg_rates))
        #print('Label Weights: {}'.format(label_weights))
        #print('Thresholds: {}'.format(thresholds))
        #print('Weights: {}'.format(weights))
        #print('Expected Diff: {:.5f}'.format(diff))

        return (diff < self.epsilon), avg_rates


class ThresholdObjective:

    def __init__(self, target: float, num_labels: int, metric_name: str):
        assert metric_name in ('max-prob', 'entropy'), 'Metric name must be `max-prob` or `entropy`'

        self._target = target
        self._num_labels = num_labels
        self._sigmoid = SharpenedSigmoid(beta=1)
        self._metric_name = metric_name
        #self._reg_weight = 0.01

    @property
    def target(self) -> float:
        return self._target

    @property
    def num_labels(self) -> int:
        return self._num_labels

    @property
    def metric_name(self) -> str:
        return self._metric_name

    def compute_metric(self, probs: np.ndarray) -> np.ndarray:
        if self.metric_name == 'entropy':
            return compute_entropy_metric(probs)
        elif self.metric_name == 'max-prob':
            return compute_max_prob_metric(probs)
        else:
            raise ValueError('Unknown metric: {}'.format(self.metric_name))

    def __call__(self, probs: np.ndarray, preds: np.ndarray, thresholds: np.ndarray, weights: np.ndarray) -> float:
        stay_probs = np.zeros(shape=(self.num_labels, ))
        stay_counts = np.zeros(shape=(self.num_labels, ))

        elevate_probs = np.zeros(shape=(self.num_labels, ))
        elevate_counts = np.zeros(shape=(self.num_labels, ))

        metrics = self.compute_metric(probs=probs)

        for metric, level_preds in zip(metrics, preds):
            first_pred, second_pred = level_preds[0], level_preds[1]

            w = np.square(weights[first_pred])
            elevation_prob = self._sigmoid(w * (thresholds[first_pred] - metric))
            stay_prob = 1.0 - elevation_prob

            stay_probs[first_pred] += stay_prob
            stay_counts[first_pred] += 1

            elevate_probs[second_pred] += elevation_prob
            elevate_counts[second_pred] += 1

        stay_rates = stay_probs / (stay_counts + SMALL_NUMBER)
        elevate_rates = elevate_probs / (elevate_counts + SMALL_NUMBER)

        # Compute the loss as a weighted average based on label frequency
        #total_count = len(preds)
        #label_weights = label_counts / total_count

        stay_loss = float(np.sum(np.abs(stay_rates - (1.0 - self.target))))
        elevate_loss = float(np.sum(np.abs(elevate_rates - self.target)))

        loss = stay_loss + elevate_loss

        return loss, stay_rates, elevate_rates, stay_counts, elevate_counts

    def derivative(self,
                   probs: np.ndarray,
                   preds: np.ndarray,
                   stay_rates: np.ndarray,
                   elevate_rates: np.ndarray,
                   stay_counts: np.ndarray,
                   elevate_counts: np.ndarray,
                   thresholds: np.ndarray,
                   weights: np.ndarray) -> np.ndarray:
        # Compute the (approx) derivative of the absolute value term for both staying and elevating
        d_stay = 2.0 * (stay_rates > (1.0 - self.target)).astype(float) - 1.0
        d_elevate = 2.0 * (elevate_rates > self.target).astype(float) - 1.0

        # Create arrays in which to accumulate gradients
        threshold_gradient = np.zeros_like(thresholds)
        weight_gradient = np.zeros_like(weights)
        num_samples = probs.shape[0]

        # Compute the per-label loss weight
        metrics = self.compute_metric(probs=probs)

        for metric, level_preds in zip(metrics, preds):
            first_pred, second_pred = level_preds[0], level_preds[1]

            # Get the weight and threshold used during this inference
            weight = weights[first_pred]
            w = np.square(weight)
            t = thresholds[first_pred]

            # Compute the derivative of the sigmoid funcion w.r.t. the input (no chain rule yet)
            diff = t - metric
            ds = self._sigmoid.derivative(w * diff)

            # Compute the gradient with respect the first prediction result (stay)
            dL_ds = (d_stay[first_pred] / stay_counts[first_pred]) * (-1 * ds)
            dL_dt = dL_ds * w
            dL_dw = dL_ds * 2 * weight * diff

            threshold_gradient[first_pred] += dL_dt
            weight_gradient[first_pred] += dL_dw

            # Compute the gradient with respect to the second prediction result (elevate)
            dL_ds = (d_elevate[second_pred] / elevate_counts[second_pred]) * ds
            dL_dt = dL_ds * w
            dL_dw = dL_ds * 2 * weight * diff

            #dL_dt = label_weights[label] * (per_label_diff[label] / label_counts[label]) * self._sigmoid.derivative(thresholds[pred] - metric)

            # Accumulate using the first prediction values, as both parameters come from the first
            # prediction (as we do not necessarily execute the second)
            threshold_gradient[first_pred] += dL_dt
            weight_gradient[first_pred] += dL_dw

        # Add the regularization loss gradient
        #dR_dw = (-2 * self._reg_weight * weight) / np.square(w)
        #weight_gradient += dR_dw

        return threshold_gradient, weight_gradient


def fit_thresholds_grad(probs: np.ndarray,
                        preds: np.ndarray,
                        target: float,
                        start_thresholds: np.ndarray,
                        start_weights: np.ndarray,
                        train_thresholds: bool,
                        train_weights: bool,
                        learning_rate: float,
                        metric_name: str) -> np.ndarray:
    rand = np.random.RandomState(seed=25893)
    num_labels = probs.shape[-1]

    thresholds = np.copy(start_thresholds)
    weights = np.copy(start_weights)
    objective = ThresholdObjective(target=target, num_labels=num_labels, metric_name=metric_name)

    # Compute the initial loss for reference
    loss, _, rates, _, _ = objective(probs=probs, preds=preds, thresholds=thresholds, weights=weights)
    step_size = learning_rate

    # Set the parameters to detect the best result
    best_loss = BIG_NUMBER
    best_thresholds = np.copy(thresholds)
    best_weights = np.copy(weights)
    best_rates = rates
    num_not_improved = 0

    # Set the gradient descent parameters
    expected_threshold_grad = np.zeros_like(thresholds)
    expected_weights_grad = np.zeros_like(weights)
    gamma = 0.9
    anneal_patience = 3

    for _ in range(MAX_ITERS):
        # Compute the loss for this iteration
        loss, stay_rates, elevate_rates, stay_counts, elevate_counts = objective(probs=probs, preds=preds, thresholds=thresholds, weights=weights)

        # Compute the gradients
        dthresholds, dweights = objective.derivative(probs=probs,
                                                     preds=preds,
                                                     stay_rates=stay_rates,
                                                     elevate_rates=elevate_rates,
                                                     stay_counts=stay_counts,
                                                     elevate_counts=elevate_counts,
                                                     thresholds=thresholds,
                                                     weights=weights)

        #approx_grad = np.zeros_like(thresholds)
        #epsilon = 1e-5

        #for i in range(num_labels):
        #    thresholds[i] += epsilon
        #    upper_loss, _, _, _, _ = objective(probs=probs, preds=preds, thresholds=thresholds, weights=weights)

        #    thresholds[i] -= 2 * epsilon
        #    lower_loss, _, _, _, _ = objective(probs=probs, preds=preds, thresholds=thresholds, weights=weights)

        #    thresholds[i] += epsilon
        #    approx_grad[i] = (upper_loss - lower_loss) / (2 * epsilon)

        ##epsilon = 1e-5

        ##weight += epsilon
        ##upper_loss, _ = objective(probs=probs, labels=labels, thresholds=thresholds, weight=weight)

        ##weight -= 2 * epsilon
        ##lower_loss, _ = objective(probs=probs, labels=labels, thresholds=thresholds, weight=weight)

        ##weight += epsilon
        ##approx_grad = (upper_loss - lower_loss) / (2 * epsilon)

        #print('True: {}'.format(dthresholds))
        #print('Approx: {}'.format(approx_grad))
        #print('==========')

        # Track the 'best' loss result
        if loss < best_loss:
            best_loss = loss
            best_thresholds = np.copy(thresholds)
            best_weights = np.copy(weights)
            num_not_improved = 0
        else:
            num_not_improved += 1

        print('Loss: {:.6f}, Best Loss: {:.6f}'.format(loss, best_loss), end='\r')

        # Update the parameters via RMSProp for next iteration
        if train_thresholds:
            expected_threshold_grad = gamma * expected_threshold_grad + (1.0 - gamma) * np.square(dthresholds)
            scaled_step_size = step_size / np.sqrt(expected_threshold_grad + SMALL_NUMBER)
            thresholds -= scaled_step_size * dthresholds

        if train_weights:
            expected_weights_grad = gamma * expected_weights_grad + (1.0 - gamma) * np.square(dweights)
            scaled_step_size = step_size / np.sqrt(expected_weights_grad + SMALL_NUMBER)
            weights -= scaled_step_size * dweights

        if (num_not_improved + 1) % anneal_patience == 0:
            step_size *= ANNEAL_RATE
            thresholds = np.copy(best_thresholds)
            weights = np.copy(best_weights)

        if num_not_improved > PATIENCE:
            print('\nConverged.')
            break

    # Compute the final loss for reporting
    best_loss, _, best_rates, _, _ = objective(probs=probs,
                                               preds=preds,
                                               thresholds=best_thresholds,
                                               weights=best_weights)

    print('Final Rates: {}'.format(best_rates))

    return best_loss, best_thresholds, best_weights, best_rates


def fit_sharpness(probs: np.ndarray,
                  labels: np.ndarray,
                  thresholds: np.ndarray,
                  start_weights: np.ndarray,
                  epsilon: float,
                  target: float,
                  metric_name: str,
                  learning_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    num_labels = np.max(labels) + 1
    weights = np.copy(start_weights)
    objective = SharpnessObjective(target=target, metric_name=metric_name, epsilon=epsilon, num_labels=num_labels)

    # Compute the initial loss for reference
    loss = objective(weights=weights)
    step_size = learning_rate

    # If the starting position does not satisfy the constraint, then stop here
    is_satisfied, rates = objective.is_satisfied(probs=probs, labels=labels, thresholds=thresholds, weights=weights)

    if not is_satisfied:
        return loss, weights, rates

    # Set the parameters to detect the best result
    best_loss = loss
    best_weights = np.copy(weights)
    num_not_improved = 0

    # Set the gradient descent parameters
    expected_weights_grad = np.zeros_like(weights)
    gamma = 0.9

    for _ in range(MAX_ITERS):
        # Compute the gradients w.r.t to the previous parameters
        dweights = objective.derivative(weights=weights)

        # Get the next `candidate` solution
        next_weights = weights + step_size * dweights

        # Evaluate this point using the loss function
        loss = objective(weights=next_weights)

        # Accept if the constraint is satisfied. Otherwise, reject and lower the learning rate
        is_satisfied, _ = objective.is_satisfied(probs=probs, labels=labels, thresholds=thresholds, weights=next_weights)

        if is_satisfied and loss > best_loss:
            weights = np.copy(next_weights)
            best_loss = loss
        else:
            step_size *= ANNEAL_RATE
            num_not_improved += 1

        print('Sharpness Loss: {:.6f}, Best Loss: {:.6f}'.format(loss, best_loss), end='\r')

        if num_not_improved >= PATIENCE:
            print('\nConverged.')
            break

    print()

    _, best_rates = objective.is_satisfied(probs=probs, labels=labels, thresholds=thresholds, weights=weights)

    return best_loss, weights, best_rates


def fit_randomization(avg_rates: np.ndarray, labels: np.ndarray, target: float, epsilon: float) -> np.ndarray:
    # Compute the label weights (for the expected value)
    num_labels = np.max(labels) + 1
    label_counts = np.bincount(labels, minlength=num_labels)
    label_weights = label_counts / np.sum(label_counts)

    # Initialize the binary search parameters
    upper = 1.0
    lower = 0.0
    rand_rate = 0.0

    while (upper > lower) and abs(upper - lower) > 1e-5:
        # Compute the expected difference
        exit_rate = (1.0 - rand_rate) * avg_rates + rand_rate * target
        abs_diff = np.abs(exit_rate - target)
        expected_diff = float(np.sum(label_weights * abs_diff))

        print('Expected Diff: {:.5f}'.format(expected_diff))

        # If the difference is under epsilon, lower the randomness. Otherwise, increase the randomness
        if expected_diff <= epsilon:
            upper = rand_rate
        else:
            lower = rand_rate

        rand_rate = (upper + lower) / 2

    return rand_rate


#def fit_randomization(probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, target: float, epsilon: float, metric_name: str) -> np.ndarray:
#    # Get the observed rates, [i, j] is the avg elevation rate for label j when predicted as i
#    num_labels = np.max(labels) + 1
#    elevation_counts = np.zeros(shape=(num_labels, num_labels))
#    total_counts = np.zeros_like(elevation_counts)
#
#    sigmoid = SharpenedSigmoid(beta=BETA)
#
#    # Compute the confidence metrics
#    if metric_name == 'max-prob':
#        metrics = compute_max_prob_metric(probs)
#    elif metric_name == 'entropy':
#        metrics = compute_entropy_metric(probs)
#    else:
#        raise ValueError('Unknown metric: {}'.format(metric_name))
#
#    # Compute the predictions for all samples
#    preds = np.argmax(probs, axis=-1)
#
#    for pred, metric, label in zip(preds, metrics, labels):
#        elevation_counts[pred, label] += sigmoid(thresholds[pred] - metric)
#        total_counts[pred, label] += 1
#
#    observed_rates = elevation_counts / (total_counts + SMALL_NUMBER)
#
#    # Get the confusion matrix, [i, j] is the number of elements classified as i that are actually j
#    # Each column sums to 1
#    confusion_mat = create_confusion_matrix(predictions=preds, labels=labels)
#    confusion_mat = confusion_mat / (np.sum(confusion_mat, axis=0, keepdims=True) + SMALL_NUMBER)
#
#    # Create the constraint coefficient matrix
#    A = (confusion_mat * (target - observed_rates)).T
#
#    # Get the constraint bounds
#    beta = np.sum(confusion_mat * observed_rates, axis=0)
#    lower = (target - epsilon) - beta
#    upper = (target + epsilon) - beta
#
#    print('Start Rates: {}'.format(beta))
#
#    # Create the 'security' distribution constraint
#    constr1 = optimize.LinearConstraint(A=A, lb=lower, ub=upper)
#
#    # Constrain the rates to the range [0, 1]
#    constr2 = optimize.LinearConstraint(A=np.eye(A.shape[0]), lb=0.0, ub=1.0)
#
#    result = optimize.minimize(fun=lambda x: np.sum(x),
#                               x0=np.ones(shape=(num_labels, )),
#                               method='trust-constr',
#                               hess=lambda x: np.zeros_like(A),
#                               constraints=[constr1, constr2],
#                               tol=SMALL_NUMBER)
#
#    print('Adjusted Rates: {}'.format(A.dot(result.x) + beta))
#    print('Rand Rates: {}'.format(result.x))
#
#    return result.x
