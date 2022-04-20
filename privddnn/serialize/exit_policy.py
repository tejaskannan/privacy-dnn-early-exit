import numpy as np

from privddnn.exiting.early_exit import EarlyExiter, MaxProbExit, RandomExit, LabelMaxProbExit
from privddnn.serialize.utils import float_to_fixed_point, array_to_fixed_point


def serialize_policy(policy: EarlyExiter, precision: int) -> str:
    lines: List[str] = []

    exit_rate = '#define EXIT_RATE {}'.format(float_to_fixed_point(policy.rates[0], 15, 16))
    lines.append(exit_rate)

    if isinstance(policy, MaxProbExit):
        threshold = '#define THRESHOLD {}'.format(float_to_fixed_point(policy.thresholds[0], precision, 16))
        lines.append(threshold)

        lines.append('#define IS_MAX_PROB')
    elif isinstance(policy, LabelMaxProbExit):
        thresholds = array_to_fixed_point(policy.thresholds, precision, 16)
        thresholds_var = 'static const int16_t THRESHOLDS[] = {{ {} }};'.format(','.join(map(str, thresholds[0])))

        lines.append(thresholds_var)
        lines.append('#define IS_LABEL_MAX_PROB')
    elif isinstance(policy, RandomExit):
        lines.append('#define IS_RANDOM')
    else:
        raise ValueError('Unknown policy with type: {}'.format(policy.name))
    
    return '\n'.join(lines)
