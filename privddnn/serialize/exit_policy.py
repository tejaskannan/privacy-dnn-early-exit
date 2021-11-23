import numpy as np

from privddnn.exiting.early_exit import EarlyExiter, MaxProbExit
from privddnn.serialize.utils import float_to_fixed_point


def serialize_policy(policy: EarlyExiter, precision: int) -> str:
    lines: List[str] = []

    exit_rate = '#define EXIT_RATE {}'.format(float_to_fixed_point(policy.rates[0], precision, 16))
    lines.append(exit_rate)

    if isinstance(policy, MaxProbExit):
        threshold = '#define THRESHOLD {}'.format(float_to_fixed_point(policy.thresholds[0], precision, 16))
        lines.append(threshold)

        lines.append('#define IS_MAX_PROB')
    else:
        raise ValueError('Unknown policy with type: {}'.format(policy.name))
    
    return '\n'.join(lines)
