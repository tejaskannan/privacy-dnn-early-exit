import numpy as np
import scipy.stats as stats
import math
from collections import namedtuple, defaultdict, Counter
from typing import Dict, DefaultDict, List, Tuple

from privddnn.utils.constants import SMALL_NUMBER
from privddnn.utils.metrics import softmax


MAX_ITERS = 150
PATIENCE = 15
NEIGHBORHOOD = 25
PRECISION = 64
TEMP_PRECISION = 64

EvalResult = namedtuple('EvalResult', ['num_correct', 'num_samples', 'level_counts', 'label_counts'])


class DataSplit:

    def __init__(self, probs: np.ndarray, labels: np.ndarray):
        self.probs = probs
        self.labels = labels
        self.original_probs = np.copy(probs)
        self.logits = np.log(np.maximum(probs, 1e-5))

    def apply_temperature(self, temp: float):
        self.probs = softmax(self.logits - temp, axis=-1)

    def revert(self):
        self.probs = self.original_probs


def n_choose_k(n: int, k: int) -> int:
    return math.factorial(n) / (math.factorial(n - k) * math.factorial(k))


def split_by_prediction(probs: np.ndarray, labels: np.ndarray) -> Dict[int, DataSplit]:
    """
    Splits the given dataset by the first-level predictions.
    """
    probs_dict: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
    labels_dict: DefaultDict[int, List[int]] = defaultdict(list)

    splits: List[DataSplit] = []
    num_labels = probs.shape[-1]

    for sample_probs, label in zip(probs, labels):
        pred = np.argmax(sample_probs[0])
        probs_dict[pred].append(np.expand_dims(sample_probs, axis=0))
        labels_dict[pred].append(label)

    results: Dict[int, DataSplit] = dict()
    for pred in sorted(probs_dict.keys()):
        split = DataSplit(probs=np.vstack(probs_dict[pred]),
                          labels=np.vstack(labels_dict[pred]))
        results[pred] = split

    return results


def eval_on_split(data_split: DataSplit, threshold: float) -> EvalResult:
    """
    Executes a Maximum Probability stopping criteria using the given threshold.
    """
    max_first_probs = np.max(data_split.probs[:, 0, :], axis=-1)  # [B]
    num_levels = np.less(max_first_probs, threshold).astype(float)  # [B]

    level_preds = np.argmax(data_split.probs, axis=-1)  # [B, L]
    preds = (1.0 - num_levels) * level_preds[:, 0] + num_levels * level_preds[:, 1]  # [B]
    preds = preds.astype(int)  # [B]

    num_correct = np.sum(np.isclose(preds, data_split.labels))
    num_samples = preds.shape[0]

    num_labels = data_split.probs.shape[-1]
    level_counts = np.zeros(shape=(num_labels, ))
    label_counts = np.zeros(shape=(num_labels, ))

    for label, num_levels in zip(data_split.labels, num_levels):
        level_counts[label] += num_levels
        label_counts[label] += 1

    return EvalResult(num_correct=num_correct, num_samples=num_samples, level_counts=level_counts, label_counts=label_counts)


def eval_on_split_rand(data_split: DataSplit, threshold: float, rand_rate: float, target: float, rand: np.random.RandomState, sampled_idx: np.ndarray) -> EvalResult:
    """
    Executes a Maximum Probability stopping criteria using the given threshold.
    """
    max_first_probs = np.max(data_split.probs[sampled_idx, 0, :], axis=-1)  # [B]
    threshold_levels = np.less(max_first_probs, threshold).astype(float)

    #threshold_perc = stats.percentileofscore(max_first_probs, score=threshold)
    #threshold_perc /= 100.0

    #lower_q_num = max(threshold_perc - target * rand_rate, 0.0)
    #upper_q_num = min(threshold_perc + (1.0 - target) * rand_rate, 1.0)

    #if abs(upper_q_num - 1.0) < SMALL_NUMBER:
    #    lower_q_num = 1.0 - rand_rate
    #elif abs(lower_q_num) < SMALL_NUMBER:
    #    upper_q_num = rand_rate

    #lower_q = np.quantile(max_first_probs, 0.25)
    #upper_q = np.quantile(max_first_probs, 0.75)

    rand_levels = np.less(rand.uniform(low=0.0, high=1.0, size=max_first_probs.shape), target).astype(float)  # [B]
    rand_mask = np.less(rand.uniform(low=0.0, high=1.0, size=max_first_probs.shape), rand_rate).astype(float)  # [B]
    #rand_mask = np.logical_or(max_first_probs < lower_q, max_first_probs > upper_q).astype(float)  # [B]

    #if rand_rate > 1e-7:
    #    print('Rand Rate: {}, True Frac: {}, Lower Q: {}, Upper Q: {}, Threshold Perc: {}'.format(rand_rate, np.sum(rand_mask) / rand_mask.shape[0], lower_q_num, upper_q_num, threshold_perc))

    num_levels = rand_mask * rand_levels + (1.0 - rand_mask) * threshold_levels

    level_preds = np.argmax(data_split.probs[sampled_idx], axis=-1)  # [B, L]
    preds = (1.0 - num_levels) * level_preds[:, 0] + num_levels * level_preds[:, 1]  # [B]
    preds = preds.astype(int)  # [B]

    labels = data_split.labels[sampled_idx]
    num_correct = np.sum(np.isclose(preds, labels))
    num_samples = preds.shape[0]

    num_labels = data_split.probs.shape[-1]
    level_counts = np.zeros(shape=(num_labels, ))
    label_counts = np.zeros(shape=(num_labels, ))

    for label, num_levels in zip(labels, num_levels):
        level_counts[label] += num_levels
        label_counts[label] += 1

    return EvalResult(num_correct=num_correct, num_samples=num_samples, level_counts=level_counts, label_counts=label_counts)


def get_stop_rates(pred_evals: Dict[int, EvalResult], num_labels: int) -> np.ndarray:
    level_counts = np.zeros(shape=(num_labels, ))
    label_counts = np.zeros(shape=(num_labels, ))

    for eval_result in pred_evals.values():
        level_counts += eval_result.level_counts
        label_counts += eval_result.label_counts

    return level_counts / (label_counts + SMALL_NUMBER)


def loss_fn(pred_evals: Dict[int, EvalResult], num_labels: int, target: float, should_print: bool = False) -> Tuple[float, int]:
    avg_level = get_stop_rates(pred_evals, num_labels=num_labels)

    diff = np.abs(avg_level - target)

    max_diff = np.zeros_like(diff)
    for i in range(len(max_diff)):
        max_diff[i] = np.max(np.abs(avg_level - avg_level[i]))

    #mean_diff = np.abs(avg_level - np.average(avg_level))
    #max_diff = np.max(diff)
    #loss = diff + nearest_diff
    #loss = diff
    #data_range = np.max(avg_level) - np.min(avg_level)
    loss = diff + max_diff

    return np.sum(loss), np.argmax(loss), avg_level


def most_freq_pred(probs: np.ndarray, labels: np.ndarray) -> List[int]:
    first_probs = probs[:, 0, :]
    num_labels = first_probs.shape[-1]
    preds = np.argmax(first_probs, axis=-1)
    correct = (preds == labels).astype(int)

    result: List[int] = []

    for label in range(num_labels):
        mask = np.equal(labels, label)
        masked_preds = np.where(mask, preds, num_labels)
        pred_counts = np.bincount(masked_preds, minlength=num_labels + 1)
        most_freq_pred = np.argmax(pred_counts[0:-1])

        #mask = (labels == label).astype(int)
        #accuracy = np.sum(correct * mask) / np.sum(mask)

        result.append(most_freq_pred)

    return result


def fit_thresholds(probs: np.ndarray, labels: np.ndarray, rate: float, start_thresholds: np.ndarray, temperature: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fits thresholds to get all labels to stop at the first level with the given rate.
    """
    assert len(start_thresholds.shape) == 1, 'Must provide a 1d thresholds array'
    assert len(temperature.shape) == 1, 'Must provide a 1d temperature array'
    num_labels = probs.shape[-1]
    target = 1.0 - rate

    # Split the dataset by prediction
    splits = split_by_prediction(probs=probs, labels=labels)

    freq_label_pred = most_freq_pred(probs, labels)

    # Scale all the probabilities by the temperature (this doesn't change later on)
    for pred, data_split in splits.items():
        data_split.apply_temperature(temperature[pred])

    # Copy the starting thresholds (so we can make non-destructive changes)
    thresholds = np.copy(start_thresholds)
    rand = np.random.RandomState(seed=148)

    # Evaluate the initial thresholds
    pred_evals: Dict[int, EvalResult] = dict()
    for pred, data_split in splits.items():
        pred_evals[pred] = eval_on_split(data_split, threshold=thresholds[pred])

    prev_loss, worst_pred, obs_rates = loss_fn(pred_evals, num_labels, target=target, should_print=True)
    pred = freq_label_pred[worst_pred]
    final_loss = prev_loss

    num_not_improved = 0
    neighborhood = 4

    for i in range(MAX_ITERS):
        best_t = thresholds[pred]
        best_loss = prev_loss

        t = best_t
        upper = 1.0
        lower = 0.0

        for _ in range(neighborhood):
            # Evaluate this threshold
            eval_result = eval_on_split(splits[pred], threshold=t)
            pred_evals[pred] = eval_result

            loss, _, avg_rates = loss_fn(pred_evals, num_labels=num_labels, target=target)

            if loss < best_loss:
                best_t = t
                best_loss = loss

            if avg_rates[pred] < target:
                lower = t
            elif avg_rates[pred] > target:
                upper = t
            else:
                break

            t = (upper + lower) / 2

        thresholds[pred] = best_t
        best_eval = eval_on_split(splits[pred], threshold=best_t)
        pred_evals[pred] = best_eval

        final_loss, worst_pred, obs_rates = loss_fn(pred_evals, num_labels=num_labels, target=target, should_print=True)

        print('Iteration: {}, Loss: {:.6f}'.format(i, prev_loss), end='\r')

        if abs(final_loss - prev_loss) < 1e-5:
            num_not_improved += 1
            pred = rand.randint(0, num_labels)
        else:
            num_not_improved = 0
            pred = freq_label_pred[worst_pred]

        prev_loss = final_loss
        neighborhood = min(neighborhood + 1, NEIGHBORHOOD)

        if num_not_improved >= PATIENCE:
            print('\nConverged')
            break

    return thresholds, final_loss, obs_rates


def fit_threshold_randomization(probs: np.ndarray, labels: np.ndarray, rate: float, thresholds: np.ndarray, epsilon: float) -> Tuple[np.ndarray, float]:
    """
    Fits thresholds to get all labels to stop at the first level with the given rate.
    """
    assert len(thresholds.shape) == 1, 'Must provide a 1d thresholds array'
    num_labels = probs.shape[-1]
    target = 1.0 - rate  # The target avg output

    # Split the dataset by prediction
    splits = split_by_prediction(probs=probs, labels=labels)

    freq_label_pred = most_freq_pred(probs, labels)

    std_devs: List[float] = []
    for pred, data_split in sorted(splits.items()):
        std = np.std(np.max(data_split.probs[:, 0, :], axis=-1))  # Scalar
        std_devs.append(std)

    # Copy the starting thresholds (so we can make non-destructive changes)
    rand = np.random.RandomState(seed=3928)
    rand_rate = 0.0

    # Evaluate the initial thresholds
    pred_evals: Dict[int, EvalResult] = dict()

    loss_sum = 0.0
    loss_count = 0.0

    upper = 1.0
    lower = 0.01
    rand_rate = 0.01

    window_size = 25
    tolerance = (1.0 / num_labels) + epsilon

    best_rate = 1.0
    best_stop_rates = np.zeros_like(thresholds)

    if abs(target) < SMALL_NUMBER or abs(1.0 - target) < SMALL_NUMBER:
        return rand_rate, 0.0, np.zeros_like(thresholds) + target

    while (upper > lower) and (upper - lower) > 1e-4:
        loss_sum = 0.0
        loss_count = 0.0

        rand_stop_rates: List[float] = []

        for _ in range(32):

            # Evaluate this threshold
            for pred, data_split in sorted(splits.items()):
                num_samples = len(data_split.probs)
                sampled_idx = rand.choice(np.arange(num_samples), size=int(0.25 * num_samples), replace=False)

                #t = rand.normal(loc=thresholds[pred], scale=(rand_rate * std_devs[pred]))
                pred_evals[pred] = eval_on_split_rand(data_split, threshold=thresholds[pred], rand_rate=rand_rate, rand=rand, target=target, sampled_idx=sampled_idx)
                #pred_evals[pred] = eval_on_split(data_split, threshold=t)

            _, _, avg_rates = loss_fn(pred_evals, num_labels=num_labels, target=target)
            rand_stop_rates.append(np.expand_dims(avg_rates, axis=0))

        stop_rates_mat = np.vstack(rand_stop_rates)
        avg_stop_rates = np.average(stop_rates_mat, axis=0)

        # Compute the probability of getting 0 with the minimum rate
        worst_prob = 0.0
        r = np.average(avg_stop_rates)
        probs = np.zeros_like(avg_stop_rates)

        for w in range(window_size + 1):
            p_w = n_choose_k(n=window_size, k=w) * np.power(r, w) * np.power(1.0 - r, window_size - w)

            p_elevate = np.power(avg_stop_rates, w)
            p_stay = np.power(1.0 - avg_stop_rates, window_size - w)
            p_w = n_choose_k(n=window_size, k=w) * np.power(r, w) * np.power(1.0 - r, window_size - w)

            denom = np.sum(p_elevate * p_stay)
            worst_prob += p_w * np.max((p_elevate * p_stay) / denom)

        #min_rate = np.min(avg_stop_rates)
        #prob_min_rate = np.power(1.0 - min_rate, window_size) / np.sum(np.power(1.0 - avg_stop_rates, window_size))

        # Compute the probabilty of getting W with the maximum rate
        #max_rate = np.max(avg_stop_rates)
        #prob_max_rate = np.power(max_rate, window_size) / np.sum(np.power(avg_stop_rates, window_size))

        #worst_prob = max(prob_min_rate, prob_max_rate)

        is_diff = (worst_prob < tolerance)

        #max_rate = np.max(avg_stop_rates)
        #min_rate = np.min(avg_stop_rates)
        #ratio = min(max_rate / min_rate, (1.0 - max_rate) / (1.0 - min_rate))

        #is_diff = ratio >= (1.0 - epsilon)

        #print('Ratio: {}, Min: {}, Max: {}'.format(ratio, min_rate, max_rate))
        #print('Avg Rates: {}'.format(avg_stop_rates))

        #print('Avg: {}'.format(np.average(stop_rates_mat, axis=0)))
        #print('Std: {}'.format(np.std(stop_rates_mat, axis=0)))

       # is_diff = True

       # for label1 in range(num_labels):
       #     dist1 = stop_rates_mat[:, label1]

       #     for label2 in range(label1 + 1, num_labels):
       #         dist2 = stop_rates_mat[:, label2]
       #         t_stat, pvalue = stats.ttest_ind(dist1, dist2, equal_var=False)

       #         #print('Label 1: {}, Label 2: {}, P Value: {}'.format(label1, label2, pvalue))

       #         if pvalue < 0.01:
       #             is_diff = False
       #             break

       #     if not is_diff:
       #         break

        print('Rand Rate: {:.5f}, Worst Prob: {:.5f}'.format(rand_rate, worst_prob), end='\n')

        if is_diff:
            upper = rand_rate

            if rand_rate < best_rate:
                best_rate = rand_rate
                best_stop_rates = avg_stop_rates
        else:
            lower = rand_rate

        rand_rate = (upper + lower) / 2

    print()
    return best_rate, 0.0, best_stop_rates
