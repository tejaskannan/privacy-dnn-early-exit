import numpy as np
from typing import List


class RandomUniformGenerator:

    def __init__(self, batch_size: int):
        self._batch_size = batch_size
        self._idx = 0
        self._vals = np.random.uniform(size=(batch_size, ))

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def reset(self):
        self._vals = np.random.uniform(size=(self.batch_size, ))
        self._idx = 0

    def get(self) -> float:
        if self._idx >= self.batch_size:
            self.reset()

        val = self._vals[self._idx]
        self._idx += 1
        return val


class RandomIntGenerator:

    def __init__(self, low: int, high: int, batch_size: int):
        self._batch_size = batch_size
        self._idx = 0
        self._low = low
        self._high = high

        self._vals = np.random.randint(low=low, high=high, size=(batch_size, ))

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def low(self) -> int:
        return self._low

    @property
    def high(self) -> int:
        return self._high

    def reset(self):
        self._vals = np.random.randint(low=self.low, high=self.high, size=(self.batch_size, ))
        self._idx = 0

    def get(self) -> int:
        if self._idx >= self.batch_size:
            self.reset()

        val = self._vals[self._idx]
        self._idx += 1
        return int(val)


class RandomChoiceGenerator:

    def __init__(self, num_choices: int, rates: List[float], batch_size: int):
        assert len(rates) == num_choices, 'Must provide same number of rates ({}) as choices ({})'.format(len(rates), num_choices)
        assert np.isclose(sum(rates), 1.0), 'The provided rates must sum to 1.'

        self._choices = list(range(num_choices))
        self._batch_size = batch_size
        self._num_choices = num_choices
        self._rates = rates
        self._idx = 0

        self._vals = np.random.choice(self._choices, size=batch_size, p=rates)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def rates(self) -> List[float]:
        return self._rates

    @property
    def num_choices(self) -> int:
        return self._num_choices

    def reset(self):
        self._vals = np.random.choice(self._choices, size=self.batch_size, p=self.rates)
        self._idx = 0

    def get(self) -> int:
        if self._idx >= self.batch_size:
            self.reset()

        val = self._vals[self._idx]
        self._idx += 1
        return int(val)
