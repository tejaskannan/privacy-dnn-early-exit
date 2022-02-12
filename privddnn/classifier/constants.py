from enum import Enum, auto


class OpName(Enum):
    LOSS = auto()
    PROBS = auto()


class ModelMode(Enum):
    TRAIN = auto()
    TEST = auto()

