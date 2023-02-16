import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import cm
from argparse import ArgumentParser
from typing import List

from privddnn.analysis.utils.read_logs import get_attack_results
from privddnn.utils.file_utils import read_json
from privddnn.utils.plotting import PLOT_STYLE, TITLE_FONT, AXIS_FONT, LEGEND_FONT, LINE_WIDTH, MARKER, MARKER_SIZE, POLICY_LABELS


matplotlib.rc('pdf', fonttype=42)
plt.rcParams['pdf.fonttype'] = 42


POLICY_COLORS = {
    'random': cm.get_cmap('Greys'),
    'max_prob': cm.get_cmap('Oranges'),
    'label_max_prob': cm.get_cmap('Purples'),
    'cgr_max_prob': cm.get_cmap('Blues'),
    'entropy': cm.get_cmap('Oranges'),
    'label_entropy': cm.get_cmap('Purples'),
    'cgr_entropy': cm.get_cmap('Blues')
}


ADJUSTMENT = 2


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--policies', type=str, required=True, nargs='+')
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    parameters = read_json(args.config)

    num_outputs_list = parameters['num_outputs']
    metric = parameters['metric']
    dataset_order = parameters['dataset_order']
    attack_model = parameters['attack_model']

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.8))

        num_datasets = len(parameters['model_configs'])
        color_indices = np.linspace(start=0.25, stop=0.75, num=len(num_outputs_list), endpoint=True)
        xs = np.arange(num_datasets)

        num_policies = len(args.policies)
        width = 1.0 / (num_policies * len(num_outputs_list) + 2)
        offset = -1 * width * (num_policies * len(num_outputs_list) - 1) / 2.0

        for policy in args.policies:
            for output_idx, num_outputs in enumerate(num_outputs_list):
                attack_scores: List[float] = []
                normalized_scores: List[float] = []
                base_normalized_scores: List[float] = []

                target_values: List[float] = []
                base_values: List[float] = []

                for model_config in parameters['model_configs']:
                    dataset_name = model_config['dataset_name']
                    test_log_folder = model_config[str(num_outputs)]

                    attack_results = get_attack_results(folder_path=test_log_folder,
                                                        fold='test',
                                                        dataset_order=dataset_order,
                                                        attack_train_log=test_log_folder,
                                                        metric=metric,
                                                        attack_model=attack_model,
                                                        attack_policy='same',
                                                        trials=1)

                    if metric == 'correct_rank':
                        aggregate = min([np.average(r) for r in attack_results[policy].values()])
                        baseline = min([np.average(r) for r in attack_results['random'].values()])
                    else:
                        aggregate = max([np.average(r) for _, r in sorted(attack_results[policy].items())])
                        baseline = max([np.average(r) for _, r in sorted(attack_results['random'].items())])

                    normalized = aggregate / baseline

                    target_values.extend([np.average(r) for r in attack_results[policy].values()])
                    base_values.extend([np.average(r) for r in attack_results['random'].values()])

                    attack_scores.append(aggregate)
                    normalized_scores.append(normalized)

                #t_stat, pvalue = stats.ttest_ind(normalized_scores, base_normalized_scores, equal_var=False)
                print('Avg normalized Result for {} on {} outputs: {:.4f}. # samples: {}'.format(policy, num_outputs, np.average(normalized_scores), len(normalized_scores)))

                color = POLICY_COLORS[policy](color_indices[output_idx])
                ax.bar(xs + offset, attack_scores, width=width, label='{}, {} Exits'.format(POLICY_LABELS[policy], num_outputs), edgecolor='k', linewidth=1, color=color)
                offset += width

        dataset_names = [config['dataset_name'] for config in parameters['model_configs']]

        ax.set_xticks(xs)
        ax.set_xticklabels(dataset_names, fontsize=LEGEND_FONT + ADJUSTMENT)

        ax.tick_params(axis='y', labelsize=LEGEND_FONT + ADJUSTMENT)
        
        ylabel = 'Attacker Accuracy (%)' if metric == 'accuracy' else 'Normalied Average Correct Rank'

        ax.set_xlabel('Dataset', fontsize=AXIS_FONT + ADJUSTMENT)
        ax.set_ylabel(ylabel, fontsize=AXIS_FONT + ADJUSTMENT)
        ax.set_title('Attack Accuracy for Varying Numbers of DNN Exits', fontsize=TITLE_FONT + ADJUSTMENT)

        ax.legend(fontsize=LEGEND_FONT + ADJUSTMENT, bbox_to_anchor=(1.0, 0.9))

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
