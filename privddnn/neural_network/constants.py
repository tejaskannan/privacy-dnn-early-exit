from enum import Enum, auto


class OpName(Enum):
    LOGITS = auto()
    PREDICTIONS = auto()
    LOSS = auto()
    OPTIMIZE = auto()
    PROBS = auto()
    STATE = auto()


class PhName(Enum):
    INPUTS = auto()
    LABELS = auto()
    DROPOUT_KEEP_RATE = auto()
    LOSS_WEIGHT = auto()
    MODEL_CORRECT = auto()
    STOP_RATES = auto()


class MetaName(Enum):
    INPUT_SHAPE = auto()
    INPUT_MEAN = auto()
    INPUT_STD = auto()
    NUM_LABELS = auto()


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
