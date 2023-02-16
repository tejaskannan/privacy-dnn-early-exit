
MARKER = 'o'
MARKER_SIZE = 8
LINE_WIDTH = 3
CAPSIZE = 5
PLOT_STYLE = 'seaborn-ticks'
AXIS_FONT = 14
TITLE_FONT = 16
LABEL_FONT = 12
LEGEND_FONT = 12

COLORS = {
    'max_prob': '#c7e9b4',
    'label_max_prob': '#51b6c4',
    'cgr_max_prob': '#2c7fb8',
    'random': 'gray',
    'entropy': '#9ebcda',
    'label_entropy': '#8c96c6',
    'cgr_entropy': '#810f7c',
    'adaptive_random_max_prob': '#2c7fb8'
}


DATASET_LABELS = {
    'emnist': 'EMNIST',
    'mnist': 'MNIST',
    'fashion_mnist': 'Fash. MNIST',
    'uci_har': 'Activity',
    'cifar10': 'Cifar10',
    'cifar100': 'Cifar100',
    'wisdm_real': 'WISDM',
    'traffic_signs': 'GTSRB',
    'speech': 'Speech'
}


POLICY_LABELS = {
    'max_prob': 'Max Prob',
    'entropy': 'Entropy',
    'label_max_prob': 'PCE Max Prob',
    'label_entropy': 'PCE Entropy',
    'cgr_max_prob': 'CGR Max Prob',
    'cgr_entropy': 'CGR Entropy',
    'random': 'Random'
}


def to_label(name: str) -> str:
    tokens = name.split('_')
    return ' '.join(t.capitalize() for t in tokens)
