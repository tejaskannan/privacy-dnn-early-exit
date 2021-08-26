from enum import Enum, auto


class OpName(Enum):
    LOGITS = auto()
    PREDICTIONS = auto()
    LOSS = auto()
    OPTIMIZE = auto()
    PROBS = auto()
    STATE = auto()
    STOP_PROBS = auto()


class PhName(Enum):
    INPUTS = auto()
    LABELS = auto()
    DROPOUT_KEEP_RATE = auto()
    LOSS_WEIGHT = auto()
    STOP_RATES = auto()
    LEARNING_RATE = auto()


class MetaName(Enum):
    INPUT_SHAPE = auto()
    NUM_LABELS = auto()
    DATASET_NAME = auto()


class ModelMode(Enum):
    TRAIN = auto()
    TEST = auto()
    FINE_TUNE = auto()
    FINE_TUNE_EVEN = auto()


# Hyperparameter keys
NUM_EPOCHS = 'num_epochs'
BATCH_SIZE = 'batch_size'
LEARNING_RATE = 'learning_rate'
LEARNING_RATE_DECAY = 'learning_rate_decay'
GRADIENT_CLIP = 'gradient_clip'
EARLY_STOP_PATIENCE = 'early_stop_patience'
DECAY_PATIENCE = 'decay_patience'
TRAIN_FRAC = 'train_frac'
DROPOUT_KEEP_RATE = 'dropout_keep_rate'
STOP_RATES = 'stop_rates'
