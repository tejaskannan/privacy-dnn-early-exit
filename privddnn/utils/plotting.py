
def to_label(name: str) -> str:
    tokens = name.split('_')
    return ' '.join(t.capitalize() for t in tokens)
