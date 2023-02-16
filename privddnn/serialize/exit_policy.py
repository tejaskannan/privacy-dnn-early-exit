import numpy as np

from privddnn.exiting.early_exit import EarlyExiter, MaxProbExit, RandomExit, LabelMaxProbExit, ConfidenceGuidedRandomExit
from privddnn.serialize.utils import float_to_fixed_point, array_to_fixed_point



def serialize_policy(policy: EarlyExiter, precision: int) -> str:
    lines: List[str] = []

    exit_rates = array_to_fixed_point(policy.rates, 15, 16)
    exit_rates_var = 'static const int16_t EXIT_RATES[] = {{ {} }};'.format(','.join(map(str, exit_rates)))
    lines.append(exit_rates_var)

    # Scale the exit rates based on the remaning levels
    scaled_exit_rates: List[float] = []
    for idx, exit_rate in enumerate(policy.rates):
        remaining = sum(policy.rates[idx:])
        scaled_exit_rates.append(exit_rate / remaining)

    scaled_exit_rates_fp = array_to_fixed_point(scaled_exit_rates, 15, 16)
    scaled_exit_rates_var = 'static const int16_t SCALED_EXIT_RATES[] = {{ {} }};'.format(','.join(map(str, scaled_exit_rates_fp)))
    lines.append(scaled_exit_rates_var)

    if isinstance(policy, MaxProbExit):
        thresholds = array_to_fixed_point(policy.thresholds, precision, 16)

        threshold = 'static const int16_t THRESHOLDS[] = {{ {} }};'.format(','.join(map(str, thresholds)))
        lines.append(threshold)

        lines.append('#define IS_MAX_PROB')
    elif isinstance(policy, LabelMaxProbExit):
        thresholds = array_to_fixed_point(policy.thresholds, precision, 16)
        thresholds_var = 'static const int16_t THRESHOLDS[] = {{ {} }};'.format(','.join(map(str, thresholds.reshape(-1))))

        lines.append(thresholds_var)
        lines.append('#define IS_LABEL_MAX_PROB')
    elif isinstance(policy, RandomExit):
        lines.append('#define IS_RANDOM')
        lines.append('static const int16_t THRESHOLDS[] = { 0 };')
    elif isinstance(policy, ConfidenceGuidedRandomExit):
        lines.append('#define IS_CGR_MAX_PROB')
        lines.append('#define MAX_BIAS {}'.format(float_to_fixed_point(policy._max_epsilon, precision, 16)))
        lines.append('#define INCREASE_FACTOR {}'.format(float_to_fixed_point(policy._increase_factor, precision, 16)))
        lines.append('#define DECREASE_FACTOR {}'.format(float_to_fixed_point(policy._decrease_factor, precision, 16)))
        lines.append('#define WINDOW_MIN {}'.format(policy._window_min))
        lines.append('#define WINDOW_MAX {}'.format(policy._window_max))
        lines.append('#define WINDOW_BITS {}'.format(int(np.log2(policy._window_max - policy._window_min + 1))))

        thresholds = array_to_fixed_point(policy.thresholds, precision, 16)
        thresholds_var = 'static const int16_t THRESHOLDS[] = {{ {} }};'.format(','.join(map(str, thresholds.reshape(-1))))
        lines.append(thresholds_var)
    else:
        raise ValueError('Unknown policy with type: {}'.format(policy.name))
    
    return '\n'.join(lines)
