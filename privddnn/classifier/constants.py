from enum import Enum, auto


class OpName(Enum):
    LOGITS = auto()
    PREDICTIONS = auto()
    LOSS = auto()
    OPTIMIZE = auto()
    PROBS = auto()
    STATE = auto()
    STOP_PROBS = auto()


class ModelMode(Enum):
    TRAIN = auto()
    TEST = auto()
    FINE_TUNE = auto()
    FINE_TUNE_EVEN = auto()

