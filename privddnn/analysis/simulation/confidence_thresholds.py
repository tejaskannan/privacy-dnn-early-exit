import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from typing import List

from privddnn.classifier import BaseClassifier, ModelMode, OpName
from privddnn.dataset import Dataset
from privddnn.exiting import ExitStrategy, EarlyExiter, make_policy, EarlyExitResult
from privddnn.restore import restore_classifier
from privddnn.utils.plotting import COLORS, MARKER, MARKER_SIZE, LINE_WIDTH, CAPSIZE, PLOT_STYLE
from privddnn.utils.plotting import AXIS_FONT, TITLE_FONT, LABEL_FONT, LEGEND_FONT, POLICY_LABELS


CLASS_ONE = 4
CLASS_TWO = 5

CLASS_ONE_NAME = 'off'
CLASS_TWO_NAME = 'on'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    # Restore the model
    model: BaseClassifier = restore_classifier(model_path=args.model_path, model_mode=ModelMode.TEST)

    # Get the predictions from the models
    val_probs = model.validate(should_approx=False)  # [B, L, K]

    # Get the validation labels (we use this to fit any policies)
    val_labels = model.dataset.get_val_labels()  # [B]

    # Make the exit policy
    rates = [0.5, 0.5]
    max_prob_policy = make_policy(strategy=ExitStrategy.MAX_PROB, rates=rates, model_path=args.model_path)
    max_prob_policy.fit(val_probs=val_probs, val_labels=val_labels)

    pce_max_prob_policy = make_policy(strategy=ExitStrategy.LABEL_MAX_PROB, rates=rates, model_path=args.model_path)
    pce_max_prob_policy.fit(val_probs=val_probs, val_labels=val_labels)

    class_one_max_probs: List[float] = []
    class_two_max_probs: List[float] = []

    for probs in val_probs:
        output_idx = pce_max_prob_policy.select_output(probs=probs)
        pred = pce_max_prob_policy.get_prediction(probs=probs, level=output_idx)
        max_prob = np.max(probs[0])

        if pred == CLASS_ONE:
            class_one_max_probs.append(max_prob)
        elif pred == CLASS_TWO:
            class_two_max_probs.append(max_prob)

    max_prob_threshold = max_prob_policy.thresholds[0]
    pce_thresholds = pce_max_prob_policy.thresholds[0]

    with plt.style.context(PLOT_STYLE):
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)

        ax0.hist(class_one_max_probs, bins=25, density=True, color='#51b6c4')
        ax0.axvline(max_prob_threshold, label='Max Prob', color='black', linestyle='--', linewidth=LINE_WIDTH)
        #ax0.axvline(pce_thresholds[CLASS_ONE], label='PCE', color='red', linestyle='--', linewidth=LINE_WIDTH)

        ax0.set_xlabel('Maximum Probability at First Exit', size=AXIS_FONT)
        ax0.set_ylabel('Density', size=AXIS_FONT)
        ax0.set_title('Confidence Values for "{}"'.format(CLASS_ONE_NAME), size=TITLE_FONT)
        ax0.legend(fontsize=LEGEND_FONT)
        ax0.tick_params(axis='both', which='major', labelsize=LABEL_FONT)

        ax1.hist(class_two_max_probs, bins=25, density=True, color='#51b6c4')
        ax1.axvline(max_prob_threshold, label='Max Prob', color='black', linestyle='--', linewidth=LINE_WIDTH)
        #ax1.axvline(pce_thresholds[CLASS_TWO], label='PCE', color='red', linestyle='--', linewidth=LINE_WIDTH)

        ax1.set_xlabel('Maximum Probability at First Exit', size=AXIS_FONT)
        ax1.set_ylabel('Density', size=AXIS_FONT)
        ax1.set_title('Confidence Values for "{}"'.format(CLASS_TWO_NAME), size=TITLE_FONT)
        ax1.legend(fontsize=LEGEND_FONT)
        ax1.tick_params(axis='both', which='major', labelsize=LABEL_FONT)

        plt.tight_layout()

        if args.output_file is not None:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
        else:
            plt.show()
