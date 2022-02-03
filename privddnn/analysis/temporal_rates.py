import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.plotting import PLOT_STYLE


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--exit-rate', type=float, required=True)
    args = parser.parse_args()

    log = read_json_gz(args.test_log)['val'][args.policy]
    exit_rate = str(round(args.exit_rate, 2))
    log_results = log[exit_rate][args.dataset_order]

    counts = np.zeros((2, 2))

    for i in range(1, len(log_results['output_levels'])):
        prev_level = log_results['output_levels'][i-1]
        curr_level = log_results['output_levels'][i]

        counts[curr_level, prev_level] += 1


   # print(log_results['output_levels'][100:125])
   # print(log_results['preds'][100:125])

    normalized_counts = counts / np.sum(counts, axis=-1, keepdims=True)
    print('Rates: {}'.format(normalized_counts))

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        x = 0
        width = 0.5

        for curr in range(2):
            for prev in range(2):
                ax.bar(x, counts[curr, prev], width=width, label='Curr {}, Prev {}'.format(curr, prev))
                x += 1

        ax.legend()
        ax.set_ylabel('Count')

        plt.show()

