
MARKER = 'o'
MARKER_SIZE = 8
LINE_WIDTH = 3
PLOT_STYLE = 'seaborn-ticks'
AXIS_FONT = 14
TITLE_FONT = 16
LABEL_FONT = 12
LEGEND_FONT = 12

COLORS = {
    'max_prob': '#c7e9b4',
    'label_max_prob': '#51b6c4',
    'hybrid_max_prob': '#225ea8',
    'random': 'black',
    'fixed': '#8856a7',
    'entropy': '#9ebcda',
    'label_entropy': '#8c96c6',
    'hybrid_entropy': '#810f7c',
    'greedy_even': '#969696',
    'even_max_prob': '#2c7fb8',
    'even_label_max_prob': '#253494',
    'buffered_max_prob': '#2c7fb8',
    'delayed_max_prob': '#2c7fb8',
    'adaptive_random_max_prob': 'red',
    'rolling_max_prob': 'gray',
    'linear': '#2c7fb8'
}


DATASET_LABELS = {
    'emnist': 'EMNIST',
    'mnist': 'MNIST',
    'fashion_mnist': 'Fashion MNIST',
    'uci_har': 'Activity'
}

def to_label(name: str) -> str:
    tokens = name.split('_')
    return ' '.join(t.capitalize() for t in tokens)
