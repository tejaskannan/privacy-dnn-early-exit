import numpy as np
import math
import scipy.stats as stats


class RandomnessController:

    def __init__(self, targets: np.ndarray, window: int, significance: float, num_labels: int):
        self._targets = targets
        self._window = window
        self._significance = significance
        self._num_labels = num_labels
        self._num_levels = 2

        self._observed = np.zeros(shape=(num_labels, self._num_levels))
        self._total = 0
        self._rand_rate = 0.0

    @property
    def window(self) -> int:
        return self._window

    @property
    def significance(self) -> float:
        return self._significance

    @property
    def targets(self) -> np.ndarray:
        return self._targets

    @property
    def num_labels(self) -> int:
        return self._num_labels

    @property
    def num_levels(self) -> int:
        return 2

    def reset(self):
        self._observed = np.zeros(shape=(self.num_labels, self.num_levels))
        self._total = 0
        self._rand_rate = 0.0

    def update(self, pred: int, level: int):
        self._observed[pred, level] += 1.0
        self._total += 1.0

    def get_rate(self, sample_idx: int) -> float:
        if (sample_idx + 1) % self.window != 0:
            return self._rand_rate

        # Run a binomial test for each predicted label where 'success' is 'staying' at level 0
        next_rate = 0.0

        for pred, observed_counts in enumerate(self._observed):
            # Skip entries with no observed values
            if np.isclose(np.sum(observed_counts), 0.0):
                continue

            pval = stats.binom_test(x=observed_counts, p=self.targets[pred], alternative='two-sided')

            if pval < self.significance:
                level0 = observed_counts[0]
                level1 = observed_counts[1]

                start_freq = level0 / (level0 + level1)

                while (next_rate < 1.0) and (pval < self.significance):
                    # Adjust the counts based on the difference direction
                    observed_freq = level0 / (level0 + level1)

                    if observed_freq < self.targets[0]:
                        level0 += 1
                        level1 -= 1
                    else:
                        level0 -= 1
                        level1 += 1

                    pval = stats.binom_test(x=[level0, level1], p=self.targets[pred], alternative='two-sided')
                    candidate_rate = (observed_freq - start_freq) / (self.targets[pred] - start_freq)

                    next_rate = max(candidate_rate, next_rate)

        self._rand_rate = max(min(next_rate, 1.0), 0.0)

        return self._rand_rate

        # The expected number of labels we should have processed
        # in the system by this step. Note this count is not the number
        # which should be elevated. Instead, it estimates the number
        # of elements in each class overall.
        #expected_counts = self._prior * (sample_idx + 1)

        # Get the "estimated" fraction of each label which got elevated
        # to the second level model
        #observed_frac = np.clip(self._observed / expected_counts, a_min=0.0, a_max=1.0) 

        #print('Observed Fracs: {}'.format(observed_frac))
        #print('Observed Counts: {}'.format(self._observed))
        #print('Expected Counts: {}'.format(expected_counts * self.target))

        #next_rate = 0.0
        #for obs_rate in observed_frac:
        #    if obs_rate < (self.target - self.epsilon):
        #        rate = (self.target - self.epsilon - obs_rate) / (self.target - obs_rate)
        #    elif obs_rate > (self.target + self.epsilon):
        #        rate = (self.target + self.epsilon - obs_rate) / (self.target - obs_rate)
        #    else:
        #        rate = 0.0

        #    next_rate = max(rate, next_rate)

        ##print('Rand Rate: {}'.format(next_rate))

        #log_prob = log_likelihood(counts=self._observed, prior=self._prior)
        #print('Log Likelihood: {}'.format(log_prob))
        #print('Lower Bound: {}'.format(math.log(1 - self.epsilon)))

        #self._rand_rate = min(max(next_rate, 0.0), 1.0)
        #return self._rand_rate
