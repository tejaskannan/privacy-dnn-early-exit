import os
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

from privddnn.utils.file_utils import read_json_gz
from privddnn.utils.plotting import to_label, COLORS, AXIS_FONT, TITLE_FONT, LEGEND_FONT
from privddnn.utils.plotting import LINE_WIDTH, MARKER, MARKER_SIZE, PLOT_STYLE
from privddnn.dataset.dataset import Dataset

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    # Get the most frequent label
    tokens = args.test_log.split(os.sep)
    dataset = Dataset(tokens[-3])
    labels = dataset.get_test_labels()
    label_freq = np.bincount(labels) / len(labels)
    most_freq = np.max(label_freq) * 100.0

    # Read the attack accuracy
    test_log = read_json_gz(args.test_log)
    attack_accuracy = test_log['attack_test']

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots()

        rates = [round(r / 10.0, 2) for r in range(11)]

        for policy_name, accuracy in attack_accuracy.items():

            print('{} & {:.5f} & {:.5f}'.format(policy_name, sum(accuracy) / len(accuracy), max(accuracy)))

            ax.plot(rates, list(map(lambda a : a * 100.0, accuracy)),
                    label=to_label(policy_name),
                    marker=MARKER,
                    markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH,
                    color=COLORS[policy_name])

        ax.axhline(y=most_freq, color='r')

        ax.set_xlabel('Frac Stopping at 2nd Output', fontsize=AXIS_FONT)
        ax.set_ylabel('Test Accuracy (%)', fontsize=AXIS_FONT)
        ax.set_title('Attack Accuracy Against Exit Policies', fontsize=TITLE_FONT)

        ax.legend(fontsize=LEGEND_FONT)

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
