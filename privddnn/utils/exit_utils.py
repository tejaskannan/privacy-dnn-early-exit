import numpy as np
from itertools import product
from typing import Tuple, List

from privddnn.utils.constants import SMALL_NUMBER


def get_adaptive_elevation_bounds(continue_rate: float, epsilon: float) -> Tuple[float, float]:
    if abs(continue_rate) < SMALL_NUMBER:
        return 0.0, 0.0
    elif abs(continue_rate - 1.0) < SMALL_NUMBER:
        return 1.0, 1.0

    max_rate = continue_rate + (1.0 - continue_rate) * epsilon
    min_rate = continue_rate * (1.0 - epsilon)

    return min_rate, max_rate


def normalize_exit_rates(rates: List[float]) -> List[float]:
    result: List[float] = []

    for idx, rate in enumerate(rates):
        rate_sum = sum(rates[idx:])
        normalized = rate / max(rate_sum, SMALL_NUMBER)
        result.append(normalized)

    return result


def get_exit_rates(single_rates: List[float], num_outputs: int) -> List[List[float]]:
    rate_settings: List[Tuple[float, ...]] = list(product(single_rates, repeat=num_outputs - 1))

    rates: List[List[float]] = []
    for setting in rate_settings:
        last_rate = round(1.0 - sum(setting), 2)
        if last_rate >= 0.0:
            rates.append(list(setting) + [last_rate])

    return rates
