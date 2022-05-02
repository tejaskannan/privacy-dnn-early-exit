import numpy as np

from privddnn.exiting.early_exit import EarlyExiter, MaxProbExit, RandomExit, LabelMaxProbExit, AdaptiveRandomExit
from privddnn.serialize.utils import float_to_fixed_point, array_to_fixed_point


def serialize_policy(policy: EarlyExiter, precision: int) -> str:
    lines: List[str] = []

    exit_rate = '#define EXIT_RATE {}'.format(float_to_fixed_point(policy.rates[0], 15, 16))
    lines.append(exit_rate)

    if isinstance(policy, MaxProbExit):
        thresholds = array_to_fixed_point(policy.thresholds, precision, 16)

        threshold = 'static const int16_t THRESHOLDS[] = {{ {} }};'.format(','.join(map(str, thresholds)))
        lines.append(threshold)

        lines.append('#define IS_MAX_PROB')
    elif isinstance(policy, LabelMaxProbExit):
        thresholds = array_to_fixed_point(policy.thresholds, precision, 16)
        thresholds_var = 'static const int16_t THRESHOLDS[] = {{ {} }};'.format(','.join(map(str, thresholds[0])))

        lines.append(thresholds_var)
        lines.append('#define IS_LABEL_MAX_PROB')
    elif isinstance(policy, RandomExit):
        lines.append('#define IS_RANDOM')
    elif isinstance(policy, AdaptiveRandomExit):
        lines.append('#define IS_ADAPTIVE_RANDOM_MAX_PROB')
        lines.append('#define MAX_BIAS {}'.format(float_to_fixed_point(policy._max_epsilon, precision, 16)))
        lines.append('#define INCREASE_FACTOR {}'.format(float_to_fixed_point(policy._increase_factor, precision, 16)))
        lines.append('#define DECREASE_FACTOR {}'.format(float_to_fixed_point(policy._decrease_factor, precision, 16)))
        lines.append('#define WINDOW_MIN {}'.format(policy._window_min))
        lines.append('#define WINDOW_MAX {}'.format(policy._window_max))
        lines.append('#define WINDOW_BITS {}'.format(int(np.log2(policy._window_max - policy._window_min + 1))))

        thresholds = array_to_fixed_point(policy.thresholds, precision, 16)
        thresholds_var = 'static const int16_t THRESHOLDS[] = {{ {} }};'.format(','.join(map(str, thresholds[0])))
        lines.append(thresholds_var)
    else:
        raise ValueError('Unknown policy with type: {}'.format(policy.name))
    
    return '\n'.join(lines)
