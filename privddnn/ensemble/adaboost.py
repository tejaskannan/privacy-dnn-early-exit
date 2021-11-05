import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from typing import Any, Dict, List

from privddnn.classifier import BaseClassifier, OpName, ModelMode
from privddnn.utils.constants import SMALL_NUMBER
from privddnn.utils.file_utils import save_pickle_gz, read_pickle_gz
from privddnn.utils.metrics import softmax, to_one_hot



class AdaBoostClassifier(BaseClassifier):

    def __init__(self, num_estimators: int, exit_size: int, clf_name: str, dataset_name: str, **kwargs: Dict[str, Any]):
        super().__init__(dataset_name=dataset_name)
        assert num_estimators > exit_size, 'Ensemble size must be greater than the exit size'
        assert num_estimators > 1, 'Must provide at least 2 estimators'

        self._num_estimators = num_estimators
        self._exit_size = exit_size
        self._clf_name = clf_name

        self._clfs = []
        for _ in range(num_estimators):
            if clf_name == 'decision_tree':
                self._clfs.append(DecisionTreeClassifier(max_depth=int(kwargs['max_depth'])))
            elif clf_name == 'logistic_regression':
                self._clfs.append(LogisticRegression(C=0.1, penalty='l2', max_iter=2500))
            else:
                raise ValueError('Unknown Classifier with name: {}'.format(clf_name))
        
        self._rand = np.random.RandomState(seed=52389)
        self._is_fit = False
        self._boost_weights = np.zeros(shape=(num_estimators, ))
        self._num_labels = 0

    @property
    def num_estimators(self) -> int:
        return self._num_estimators

    @property
    def exit_size(self) -> int:
        return self._exit_size

    def fit(self):
        # Normalize the data
        self.dataset.fit_normalizer(is_global=False)
        self.dataset.normalize_data()

        inputs = self.dataset.get_train_inputs()
        labels = self.dataset.get_train_labels()

        assert len(inputs.shape) == 2, 'Must provide 2d inputs'
        assert len(labels.shape) == 1, 'Must provide 1d labels'
        assert inputs.shape[0] == labels.shape[0], 'Misaligned inputs and labels'

        # Initialize the data weights
        num_samples = inputs.shape[0]
        num_labels = np.amax(labels) + 1
        weights = np.ones(shape=(num_samples, )) / num_samples

        for idx, clf in enumerate(self._clfs):
            print('Fitting model {}/{}'.format(idx + 1, len(self._clfs)), end='\r')

            # Fit the classifier according to the current data weights
            clf.fit(inputs, labels, sample_weight=weights)

            # Compute the (weighted) error rate
            preds = clf.predict(inputs)
            is_correct = np.isclose(preds, labels).astype(float)
            error_rate = np.sum(weights * (1.0 - is_correct)) / np.sum(weights)

            # Compute the boosting weight
            alpha = np.log((1.0 - error_rate) / (error_rate + SMALL_NUMBER)) + np.log(num_labels - 1)
            self._boost_weights[idx] = alpha

            # Reset the data weights
            weights = weights * np.exp(alpha * (1.0 - is_correct))
            weights /= np.sum(weights)

        print()
        self._num_labels = num_labels
        self._is_fit = True

    def predict_sample(self, inputs: np.ndarray, level: int) -> np.ndarray:
        """
        Gets the predicted probabilities for the given input using the
        specified model level.
        """
        assert level in (0, 1), 'The model level must be either 0 or 1'

        num_classifiers = len(self._clfs) if level == 1 else self.exit_size
        probs = np.zeros(shape=(1, self._num_labels))
        expanded_inputs = np.expand_dims(inputs, axis=0)  # [1, D]

        for idx in range(num_classifiers):
            preds = self._clfs[idx].predict(expanded_inputs)  # [1]
            one_hot = to_one_hot(preds, num_labels=self._num_labels)  # [1, K]
            boost_weight = self._boost_weights[idx]

            weighted_preds = boost_weight * one_hot
            probs += weighted_preds

        return softmax(probs, axis=-1)

        # Normalize the probabiltiies. We avoid non-linearties here for better numerical stability on the MSP430
        #probs = probs / np.sum(probs, axis=-1, keepdims=True)
        #return probs.reshape(-1)

    def validate(self, op: OpName) -> np.ndarray:
        #assert op in (OpName.PROBS, OpName.PREDICTIONS), 'Operation must be either probs or predictions. Got: {}'.format(op)
        probs = self.predict_proba(inputs=self.dataset.get_val_inputs())

        if op == OpName.PREDICTIONS:
            return np.argmax(probs, axis=-1)

        return probs

    def test(self, op: OpName) -> np.ndarray:
        #assert op in (OpName.PROBS, OpName.PREDICTIONS), 'Operation must be either probs or predictions. Got: {}'.format(op)
        probs = self.predict_proba(inputs=self.dataset.get_test_inputs())

        if op == OpName.PREDICTIONS:
            return np.argmax(probs, axis=-1)

        return probs

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        assert len(inputs.shape) == 2, 'Must provide 2d inputs'
        assert self._is_fit, 'Must call fit() first'

        num_samples = inputs.shape[0]
        first_level_probs = np.ones(shape=(num_samples, self._num_labels))  # [N, K]
        second_level_probs = np.ones(shape=(num_samples, self._num_labels))

        for idx, clf in enumerate(self._clfs):
            preds = clf.predict(inputs)  # [N]
            one_hot = to_one_hot(preds, num_labels=self._num_labels)  # [N, K]
            boost_weight = self._boost_weights[idx]

            weighted_preds = boost_weight * one_hot

            if idx < self.exit_size:
                first_level_probs += weighted_preds

            second_level_probs += weighted_preds

        # Normalize the weights to create a 'probability' distribution
        #first_level_probs = first_level_probs / np.sum(first_level_probs, axis=-1, keepdims=True)  # [N, K]
        #first_level_probs = np.expand_dims(first_level_probs, axis=1)  # [N, 1, K]
        
        #second_level_probs = second_level_probs / np.sum(second_level_probs, axis=-1, keepdims=True)  # [N, K]
        #second_level_probs = np.expand_dims(second_level_probs, axis=1)  # [N, 1, K]

        first_level_probs = np.expand_dims(softmax(first_level_probs, axis=-1), axis=1)  # [N, 1, K]
        second_level_probs = np.expand_dims(softmax(second_level_probs, axis=-1), axis=1)  # [N, 1, K]

        return np.concatenate([first_level_probs, second_level_probs], axis=1)  # [N, 2, K]

    def save(self, path: str):
        serialized = {
            'dataset_name': self.dataset.dataset_name,
            'boost_weights': self._boost_weights,
            'classifiers': self._clfs,
            'classifier_name': self._clf_name,
            'num_estimators': self.num_estimators,
            'exit_size': self.exit_size,
            'is_fit': self._is_fit,
            'num_labels': self._num_labels
        }

        save_pickle_gz(serialized, path)

    @classmethod
    def restore(cls, path: str, model_mode: ModelMode):
        serialized = read_pickle_gz(path)

        clf_name = serialized['classifier_name']
        max_depth = serialized['classifiers'][0].get_params()['max_depth'] if (clf_name == 'decision_tree') else 0

        model = cls(num_estimators=serialized['num_estimators'],
                    exit_size=serialized['exit_size'],
                    clf_name=clf_name,
                    dataset_name=serialized['dataset_name'],
                    max_depth=max_depth)

        model._clfs = serialized['classifiers']
        model._boost_weights = serialized['boost_weights']
        model._is_fit = serialized['is_fit']
        model._num_labels = serialized['num_labels']

        # Normalize the data
        model.dataset.fit_normalizer(is_global=False)
        model.dataset.normalize_data()

        return model
