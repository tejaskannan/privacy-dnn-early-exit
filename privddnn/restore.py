from privddnn.classifier import BaseClassifier, ModelMode
from privddnn.neural_network import restore_model
from privddnn.ensemble.adaboost import AdaBoostClassifier


def restore_classifier(model_path: str, model_mode: ModelMode) -> BaseClassifier:
    if ('decision_tree' in model_path) or ('logistic_regression' in model_path):
        return AdaBoostClassifier.restore(path=model_path, model_mode=model_mode)
    else:
        return restore_model(path=model_path, model_mode=model_mode)
