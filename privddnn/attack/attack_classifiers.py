import numpy as np
import tensorflow as tf2
import tensorflow.compat.v1 as tf1
from collections import defaultdict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, List, DefaultDict


MAJORITY = 'Majority'
MOST_FREQ = 'MostFrequent'
LOGISTIC_REGRESSION = 'LogisticRegression'
NAIVE_BAYES = 'NaiveBayes'
NGRAM = 'Ngram'


ACCURACY = 'accuracy'
TOP2 = 'top2'


class AttackClassifier:

    def name(self) -> str:
        raise NotImplementedError()

    def fit(self, inputs: np.ndarray, labels: np.ndarray):
        raise NotImplementedError()

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> List[int]:
        raise NotImplementedError()

    def predict(self, inputs: np.ndarray) -> int:
        return self.predict_rankings(inputs=inputs, top_k=1)[0]

    def score(self, inputs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the classifier on the given dataset.

        Args:
            inputs: A [B, D] array of input features (D) for each sample (B)
            labels: A [B] array of data labels.
        Returns:
            A dictionary of score metric name -> metric value
        """
        assert len(inputs.shape) == 2, 'Must provide a 2d array of inputs'
        assert len(labels.shape) == 1, 'Must provide a 1d array of labels'
        assert inputs.shape[0] == labels.shape[0], 'Inputs and Labels are misaligned'

        correct_count = 0.0
        top2_count = 0.0
        total_count = 0.0

        for count, label in zip(inputs, labels):
            preds = self.predict_rankings(count, top_k=2)

            top2_count += float(label in preds)
            correct_count += float(preds[0] == label)
            total_count += 1.0

        return {
            ACCURACY: correct_count / total_count,
            TOP2: top2_count / total_count
        }


class MajorityClassifier(AttackClassifier):

    def __init__(self):
        self._clf: Dict[int, List[int]] = dict()
        self._most_freq = 0

    @property
    def name(self) -> str:
        return MAJORITY

    def fit(self, inputs: np.ndarray, labels: np.ndarray):
        """
        Fits the majority classifier which maps labels to the most
        frequent label for each level count.

        Args:
            inputs: A [B, D] array of input features (D) for each input sample (B)
            labels: A [B] array of data labels for each input sample (B)
        """
        assert len(inputs.shape) == 2, 'Must provide a 2d array of inputs'
        assert len(labels.shape) == 1, 'Must provide a 1d array of labels'
        assert inputs.shape[0] == labels.shape[0], 'Inputs and Labels are misaligned'

        label_counts: DefaultDict[int, List[int]] = defaultdict(list)
        num_labels = np.max(labels) + 1

        for input_features, label in zip(inputs, labels):
            count = np.sum(input_features)
            label_counts[count].append(label)

        for count, count_labels in sorted(label_counts.items()):
            freq = np.bincount(count_labels, minlength=num_labels)
            most_freq = np.argsort(freq)[::-1]
            self._clf[count] = most_freq

        label_counts = np.bincount(labels, minlength=num_labels)
        self._most_freq = np.argsort(label_counts)[::-1]

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> List[int]:
        assert len(inputs.shape) == 1, 'Must provide a 1d array of input features'
        count = np.sum(inputs)
        rankings = self._clf.get(count, self._most_freq)
        return rankings[0:top_k].astype(int).tolist()


class NgramClassifier(AttackClassifier):

    def __init__(self):
        self._clf: Dict[Tuple[int, ...], np.ndarray] = dict()
        self._most_freq = 0

    @property
    def name(self) -> str:
        return NGRAM

    def fit(self, inputs: np.ndarray, labels: np.ndarray):
        """
        Fits the majority classifier which maps labels to the most
        frequent label for each level count.

        Args:
            inputs: A [B, D] array of input features (D) for each input sample (B)
            labels: A [B] array of data labels for each input sample (B)
        """
        assert len(inputs.shape) == 2, 'Must provide a 2d array of inputs'
        assert len(labels.shape) == 1, 'Must provide a 1d array of labels'
        assert inputs.shape[0] == labels.shape[0], 'Inputs and Labels are misaligned'

        label_counts: DefaultDict[Tuple[int, ...], List[int]] = defaultdict(list)
        num_labels = np.max(labels) + 1

        for input_features, label in zip(inputs, labels):
            features = tuple(input_features.astype(int).tolist())
            label_counts[features].append(label)

        for features, feature_labels in sorted(label_counts.items()):
            freq = np.bincount(feature_labels, minlength=num_labels)
            most_freq = np.argsort(freq)[::-1]
            self._clf[features] = most_freq

        label_counts = np.bincount(labels, minlength=num_labels)
        self._most_freq = np.argsort(label_counts)[::-1]

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> List[int]:
        assert len(inputs.shape) == 1, 'Must provide a 1d array of input features'
        features = tuple(inputs.astype(int).tolist())
        rankings = self._clf.get(features, self._most_freq)
        return rankings[0:top_k].astype(int).tolist()



class MostFrequentClassifier(AttackClassifier):

    def __init__(self, num_labels: int, window: int):
        self._clf: Dict[int, np.array] = dict()  # Maps output level to an array of predictions ordered by frequency
        self._num_labels = num_labels
        self._window = window

    @property
    def name(self) -> str:
        return MOST_FREQ

    @property
    def num_labels(self) -> int:
        return self._num_labels

    @property
    def window(self) -> int:
        return self._window

    def fit(self, inputs: np.ndarray, labels: np.ndarray):
        """
        Fits the classifier by finding the most frequent
        prediction for each output level.

        Args:
            inputs: Unused (for compatability reasons)
            labels: A [B, K] array of predictions for each sample (B) and level (K)
        Returns:
            Nothing. The object saves the results internally.
        """
        assert len(labels.shape) == 2, 'Must provide a 2d array of validation predictions'

        # Unpack the shape
        num_samples, num_levels = labels.shape

        for level in range(num_levels):
            level_preds = np.bincount(labels[:, level], minlength=self.num_labels)
            self._clf[level] = np.argsort(level_preds)[::-1]

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> List[int]:
        count = np.sum(inputs)
        level = int(count >= (self.window / 2.0))
        rankings = self._clf[level]
        return rankings[0:top_k].astype(int).tolist()


class SklearnClassifier(AttackClassifier):

    def __init__(self):
        self._scaler = StandardScaler()

    def fit(self, inputs: np.ndarray, labels: np.ndarray):
        """
        Fits a classifier for the given dataset.
        """
        scaled_inputs = self._scaler.fit_transform(inputs)
        self._clf.fit(scaled_inputs, labels)

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> List[int]:
        scaled_inputs = self._scaler.transform(np.expand_dims(inputs, axis=0))
        probs = self._clf.predict_proba(scaled_inputs)[0]  # [L]
        rankings = np.argsort(probs)[::-1]
        return rankings[0:top_k].astype(int).tolist()

    def score(self, inputs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the classifier on the given dataset.

        Args:
            inputs: A [B, D] array of input features (D) for each sample (B)
            labels: A [B] array of data labels.
        Returns:
            A dictionary of score metric name -> metric value
        """
        assert len(inputs.shape) == 2, 'Must provide a 2d array of inputs'
        assert len(labels.shape) == 1, 'Must provide a 1d array of labels'
        assert inputs.shape[0] == labels.shape[0], 'Inputs and Labels are misaligned'

        scaled_inputs = self._scaler.transform(inputs)
        probs = self._clf.predict_proba(scaled_inputs)  # [B, L]

        accuracy = accuracy_score(y_true=labels, y_pred=np.argmax(probs, axis=-1))
        top2 = top_k_accuracy_score(y_true=labels, y_score=probs, k=2)

        return {
            ACCURACY: float(accuracy),
            TOP2: float(top2)
        }


class LogisticRegressionClassifier(SklearnClassifier):

    def __init__(self):
        super().__init__()
        self._clf = LogisticRegression(random_state=243089, max_iter=1000)

    @property
    def name(self) -> str:
        return LOGISTIC_REGRESSION


class NaiveBayesClassifier(SklearnClassifier):

    def __init__(self):
        super().__init__()
        self._clf = BernoulliNB(alpha=1.0, fit_prior=True)

    @property
    def name(self) -> str:
        return NAIVE_BAYES
