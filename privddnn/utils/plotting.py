
MARKER = 'o'
MARKER_SIZE = 8
LINE_WIDTH = 3
PLOT_STYLE = 'seaborn-ticks'
AXIS_FONT = 14
TITLE_FONT = 16
LABEL_FONT = 12
LEGEND_FONT = 12

COLORS = {
    'max_prob': '#a1dab4',
    'label_max_prob': '#41b6c4',
    'hybrid_max_prob': '#225ea8',
    'random': 'black',
    'entropy': '#9ebcda',
    'label_entropy': '#8c96c6',
    'hybrid_entropy': '#810f7c',
    'most_freq': 'red'
}

def to_label(name: str) -> str:
    tokens = name.split('_')
    return ' '.join(t.capitalize() for t in tokens)
