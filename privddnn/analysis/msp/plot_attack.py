import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple

from privddnn.attack.attack_classifiers import DecisionTreeEnsembleCount
from privddnn.msp_server.analyze_packet_trace import extract_message_sizes, classify_decisions
from privddnn.utils.file_utils import read_jsonl_gz
from privddnn.utils.plotting import PLOT_STYLE, TITLE_FONT, AXIS_FONT, LEGEND_FONT, LABEL_FONT, POLICY_LABELS, COLORS, DATASET_LABELS


matplotlib.rc('pdf', fonttype=42)
plt.rcParams['pdf.fonttype'] = 42


POLICIES = ['random', 'max_prob', 'label_max_prob', 'cgr_max_prob']
ADJUSTMENT = 2


def make_exit_features(exit_decisions: List[int], model_preds: List[int], window_size: int, num_labels: int) -> Tuple[np.ndarray, np.ndarray]:
    exit_decision_blocks: List[np.ndarray] = []
    labels: List[int] = []

    for start_idx in range(0, len(exit_decisions), window_size):
        end_idx = start_idx + window_size

        # Make the input features
        decision_block = np.zeros(shape=(window_size, 2))

        for offset in range(clf.window_size):
            exit_decision = exit_decisions[min(start_idx + offset, len(exit_decisions) - 1)]
            decision_block[offset, exit_decision] = 1

        exit_decision_blocks.append(np.expand_dims(decision_block, axis=0))

        pred_counts = np.bincount(model_preds[start_idx:end_idx], minlength=num_labels)
        labels.append(int(np.argmax(pred_counts)))

    return np.vstack(exit_decision_blocks), np.vstack(labels).reshape(-1)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--msp-results-folder', type=str, required=True, help='Folder containing the MSP results.')
    parser.add_argument('--attack-model-folder', type=str, required=True, help='Folder containing the attack models per policy.')
    parser.add_argument('--output-file', type=str, help='Path in which to save the resulting plot.')
    args = parser.parse_args()

    attack_accuracies: List[float] = []
    inference_accuracies: List[float] = []
    
    font_size = TITLE_FONT + ADJUSTMENT
    rand = np.random.RandomState(seed=58103)

    for policy_name in POLICIES:
        packet_trace_path = os.path.join(args.msp_results_folder, '{}.csv'.format(policy_name)) 
        message_sizes, _ = extract_message_sizes(packet_trace_path, rand=rand)

        exit_decisions = classify_decisions(message_sizes)

        server_trace_path = os.path.join(args.msp_results_folder, '{}.jsonl.gz'.format(policy_name))
        server_results: List[Dict[str, int]] = list(read_jsonl_gz(server_trace_path))
        model_preds = [int(record['pred']) for record in server_results]

        attack_model_path = os.path.join(args.attack_model_folder, '{}-attack-model.pkl.gz'.format(policy_name))
        clf = DecisionTreeEnsembleCount.restore(attack_model_path)

        exit_decision_features, exit_decision_labels = make_exit_features(exit_decisions=exit_decisions,
                                                                          model_preds=model_preds,
                                                                          window_size=clf.window_size,
                                                                          num_labels=clf.num_labels)

        attack_results = clf.score(inputs=exit_decision_features, labels=exit_decision_labels)

        true_labels: List[int] = [int(record['label']) for record in server_results]
        inference_accuracy = accuracy_score(y_true=true_labels, y_pred=model_preds)

        attack_accuracies.append(100.0 * attack_results['accuracy'])
        inference_accuracies.append(100.0 * inference_accuracy)

    # Get the dataset name from the provided paths
    path_tokens = args.attack_model_folder.split(os.sep)
    dataset_name = path_tokens[-3] if len(path_tokens[-1]) > 0 else path_tokens[-4]

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))

        xs = np.arange(2)
        width = 0.2
        offset = -width * (len(POLICIES) - 1) / 2

        for idx, (policy, inference_accuracy, attack_accuracy) in enumerate(zip(POLICIES, inference_accuracies, attack_accuracies)):
            if idx == 0:
                xoffset = -0.17
                yoffset = 2.0
            elif idx == 1:
                xoffset = -0.14
                yoffset = 2.0
            elif idx == 2:
                xoffset = -0.1
                yoffset = 2.0
            else:
                xoffset = -0.07
                yoffset = 2.0

            ax.annotate('{:.2f}'.format(inference_accuracy), (xs[0] + offset, inference_accuracy), (xs[0] + offset + xoffset, inference_accuracy + yoffset), fontsize=TITLE_FONT)

            if idx == 0:
                xoffset = -0.14
                yoffset = 2.0
            elif idx == 1:
                xoffset = -0.09
                yoffset = 2.0
            elif idx == 2:
                xoffset = -0.09
                yoffset = 5.0
            else:
                xoffset = -0.09
                yoffset = 2.0

            ax.annotate('{:.2f}'.format(attack_accuracy), (xs[1] + offset, attack_accuracy), (xs[1] + offset + xoffset, attack_accuracy + yoffset), fontsize=TITLE_FONT)

            ax.bar(xs + offset, [inference_accuracy, attack_accuracy], width=width, label=POLICY_LABELS[policy], edgecolor='black', linewidth=1, color=COLORS[policy])
            offset += width

        ax.legend(fontsize=TITLE_FONT, bbox_to_anchor=(0.5, 0.5))

        ax.set_xticks(xs)
        ax.set_xticklabels(['Inference Accuracy', 'Attack Accuracy'], fontsize=font_size)

        ax.set_ylim(bottom=0, top=110)

        yticks = list(map(int, ax.get_yticks()))
        ax.set_yticklabels(yticks, fontsize=font_size)

        ax.set_ylabel('Accuracy (%)', fontsize=font_size)
        ax.set_title('MCU Results for the {} Dataset'.format(DATASET_LABELS[dataset_name]), fontsize=TITLE_FONT + ADJUSTMENT)

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
