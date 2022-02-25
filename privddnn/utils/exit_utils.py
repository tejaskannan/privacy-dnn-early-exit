import numpy as np
from typing import Tuple


def get_adaptive_elevation_bounds(continue_rate: float, epsilon: float) -> Tuple[float, float]:
    if np.isclose(continue_rate, 0.0):
        return 0.0, 0.0
    elif np.isclose(continue_rate, 1.0):
        return 1.0, 1.0

    max_rate = continue_rate + (1.0 - continue_rate) * epsilon
    min_rate = continue_rate * (1.0 - epsilon)

    return min_rate, max_rate


def triangle_wave(step: int, window: int, amplitude: float):
    step_mod = step % window
    midpoint = window / 2.0

    if step_mod <= midpoint:
        normalized = step_mod / midpoint
        return amplitude * (1.0 - normalized)
    else:
        normalized = (step_mod - midpoint) / midpoint
        return amplitude * normalized
