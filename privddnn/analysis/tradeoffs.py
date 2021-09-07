import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from privddnn.utils.file_utils import read_json
from privddnn.utils.plotting import to_label


COLORS = {
    'max_prob': '#a1dab4',
    'label_max_prob': '#41b6c4',
    'optimized_max_prob': '#225ea8',
    'random': 'black'
}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-log', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    test_log = read_json(args.test_log)
    policies = list(test_log['accuracy'].keys())

    with plt.style.context('seaborn-ticks'):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))

        for policy_name in policies:
            rates = test_log['observed_rates'][policy_name]
            accuracy = test_log['accuracy'][policy_name]
            mut_info = test_log['mutual_information'][policy_name]
            attack_accuracy = test_log['attack_accuracy'][policy_name]

            print('{} & {:.5f} & {:.5f} & {:.5f}'.format(policy_name, np.average(accuracy), np.max(mut_info), np.max(attack_accuracy)))

            ax1.plot(rates, accuracy, marker='o', markersize=8, linewidth=3, label=to_label(policy_name), color=COLORS[policy_name])
            ax2.plot(rates, mut_info, marker='o', markersize=8, linewidth=3, label=to_label(policy_name), color=COLORS[policy_name])

        ax1.set_xlabel('Frac stopping at 2nd Output', fontsize=14)
        ax1.set_ylabel('Accuracy', fontsize=14)
        ax1.set_title('Model Accuracy', fontsize=16)
        ax1.legend(fontsize=12)
        ax1.tick_params(axis='both', which='major', labelsize=12)

        ax2.set_xlabel('Frac stopping at 2nd Output', fontsize=14)
        ax2.set_ylabel('Mutual Information', fontsize=14)
        ax2.set_title('Mut Info between Label and Output #', fontsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=12)

        plt.tight_layout()

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
