import numpy as np
from typing import Tuple


def get_adaptive_elevation_bounds(target_elevation_rate: float, epsilon: float) -> Tuple[float, float]:
    if np.isclose(target_elevation_rate, 0.0):
        return 0.0, 0.0
    elif np.isclose(target_elevation_rate, 1.0):
        return 1.0, 1.0

    max_rate = target_elevation_rate + (1.0 - target_elevation_rate) * epsilon
    min_rate = target_elevation_rate - target_elevation_rate * epsilon

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

    #min_rate = target_elevation_rate * (1.0 - epsilon)
    #max_rate = min(target_elevation_rate * (1.0 + epsilon), target_elevation_rate + (1.0 - target_elevation_rate) / 2.0)

    #if np.isclose(target_elevation_rate, 0.0) or np.isclose(target_elevation_rate, 1.0):
    #    return min_rate, max_rate

    #min_rate_gap = target_elevation_rate - min_rate
    #max_rate_gap = max_rate - target_elevation_rate

    #if max_rate_gap <= min_rate_gap:
    #    min_rate = target_elevation_rate * (1.0 - max_rate) / (1.0 - target_elevation_rate)
    #else:
    #    max_rate = 1.0 - min_rate * (1.0 - target_elevation_rate) /  target_elevation_rate

    #return min_rate, max_rate

#    min_rate_gap = target_elevation_rate - min_rate
#    max_rate_gap = min_rate_gap * target_elevation_rate / (1.0 - target_elevation_rate)
#    max_rate = target_elevation_rate + max_rate_gap
#
#    cutoff = target_elevation_rate + (1.0 - target_elevation_rate) / 2.0
#    if max_rate > cutoff:
#        max_rate = cutoff
#        max_rate_gap = max_rate - target_elevation_rate
#        min_rate_gap = max_rate_gap * (1.0 - target_elevation_rate) / target_elevation_rate
#        min_rate = target_elevation_rate - min_rate_gap
#
#    return min_rate, max_rate


#for rate in np.arange(0.0, 1.0, 0.05):
#    print(rate)
#    print(get_adaptive_elevation_bounds(target_elevation_rate=rate, epsilon=0.1))
