import numpy as np
from typing import Tuple

from privddnn.utils.constants import SMALL_NUMBER, BIG_NUMBER
from privddnn.utils.metrics import create_confusion_matrix


MAX_ITER = 500


class RandomizedObjective:

    def __init__(self, confusion_mat: np.ndarray, target: float):
        self._confusion_mat = confusion_mat
        self._num_labels = self._confusion_mat.shape[0]
        self._target = target

    def __call__(self, rand_rates: np.ndarray, should_print: bool = False) -> Tuple[float, np.ndarray]:
        # Get the probability of elevation for each (predicted) label
        prob_elevation = np.zeros(shape=(self._num_labels, ))
        elevated_counts = self._confusion_mat.T.dot(rand_rates)
        stay_counts = np.sum(self._confusion_mat, axis=-1)

        for pred in range(self._num_labels):
            num_elevated = elevated_counts[pred]
            num_pred_first = stay_counts[pred]
            denom = num_elevated + num_pred_first * (1.0 - rand_rates[pred])

            prob_elevation[pred] = num_elevated / (denom + SMALL_NUMBER)

        loss = float(0.5 * np.sum(np.square(prob_elevation - self._target)))
        return loss, prob_elevation

    def derivative(self, rand_rates: np.ndarray, prob_elevation: np.ndarray) -> np.ndarray:
        dL = (prob_elevation - self._target)
        elevated_counts = self._confusion_mat.T.dot(rand_rates)
        stay_counts = np.sum(self._confusion_mat, axis=-1)

        gradient = np.zeros_like(rand_rates)

        for rand_idx in range(self._num_labels):
            for label_idx in range(self._num_labels):
                numerator0 = self._confusion_mat[rand_idx, label_idx] * (elevated_counts[label_idx] + stay_counts[label_idx] * (1.0 - rand_rates[label_idx]))
                numerator1 = elevated_counts[label_idx] * (self._confusion_mat[rand_idx, label_idx] - int(rand_idx == label_idx) * stay_counts[label_idx])
                denom = np.square(elevated_counts[label_idx] + stay_counts[label_idx] * (1.0 - rand_rates[label_idx]))

                gradient[rand_idx] += dL[label_idx] * ((numerator0 - numerator1) / (denom + SMALL_NUMBER))

        return gradient


def fit_even_randomization(preds: np.ndarray, target: float, num_labels: int) -> float:
    confusion_mat = create_confusion_matrix(predictions=preds[:, 0], labels=preds[:, 1]) + 1
    rand = np.random.RandomState(seed=53489)
    rand_rates = rand.uniform(low=target - 0.05, high=target + 0.05, size=num_labels)

    best_loss = BIG_NUMBER
    step_size = 1e-1

    for _ in range(MAX_ITER):
        objective = RandomizedObjective(target=target, confusion_mat=confusion_mat)
        loss, prob_elevation = objective(rand_rates=rand_rates)

        if loss < best_loss:
            best_loss = loss
            best_rates = np.copy(rand_rates)

        drates = objective.derivative(rand_rates=rand_rates, prob_elevation=prob_elevation)
        rand_rates = rand_rates - step_size * drates

        print('Loss: {:.6f}'.format(loss), end='\r')

    print()

    final_loss, prob_elevation = objective(rand_rates=best_rates, should_print=True)
    print(prob_elevation)

    # Run one evaluation the best rates to confirm they are acting (about) correctly
    elevation_counts = np.zeros_like(rand_rates)
    total_counts = np.zeros_like(rand_rates)

    for _ in range(50):
        for sample_preds in preds:
            first_pred = sample_preds[0]
            rand_rate = rand_rates[first_pred]

            if np.random.uniform() < rand_rate:
                elevation_counts[sample_preds[1]] += 1
                total_counts[sample_preds[1]] += 1
            else:
                total_counts[sample_preds[0]] += 1

    elevation_rates = elevation_counts / total_counts
    print('Elevation Rates: {}'.format(elevation_rates))

    #epsilon = 1e-5
    #approx_grad = np.zeros_like(drates)

    #for idx in range(num_labels):
    #    rand_rates[idx] += epsilon
    #    upper_loss, upper_probs = objective(rand_rates=rand_rates)

    #    rand_rates[idx] -= 2 * epsilon
    #    lower_loss, lower_probs = objective(rand_rates=rand_rates)

    #    #print('Probs Grad: {}'.format(np.sum((upper_probs - lower_probs) / (2 * epsilon))))

    #    rand_rates[idx] += epsilon
    #    approx_grad[idx] = (upper_loss - lower_loss) / (2 * epsilon)

    #print('Approx Grad: {}'.format(approx_grad))



