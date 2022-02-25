import numpy as np
from collections import defaultdict, Counter
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, List, DefaultDict

from privddnn.utils.constants import BIG_NUMBER


MAJORITY = 'Majority'
MOST_FREQ = 'MostFrequent'
LOGISTIC_REGRESSION = 'LogisticRegression'
NAIVE_BAYES = 'NaiveBayes'
NGRAM = 'Ngram'
RATE = 'Rate'


ACCURACY = 'accuracy'
TOP2 = 'top2'
TOP5 = 'top5'
TOP10 = 'top10'
TOP_ALL_BUT_ONE = 'top(k-1)'


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
        top5_count = 0.0
        top10_count = 0.0
        toplast_count = 0.0
        total_count = 0.0

        num_labels = np.amax(labels) + 1

        for count, label in zip(inputs, labels):
            preds = self.predict_rankings(count, top_k=2)
            top5_preds = self.predict_rankings(count, top_k=5)
            top10_preds = self.predict_rankings(count, top_k=10)
            toplast_preds = self.predict_rankings(count, top_k=num_labels - 1)

            top2_count += float(label in preds)
            top5_count += float(label in top5_preds)
            top10_count += float(label in top10_preds)
            toplast_count += float(label in toplast_preds)
            correct_count += float(preds[0] == label)
            total_count += 1.0

        return {
            ACCURACY: correct_count / total_count,
            TOP2: top2_count / total_count,
            TOP5: top5_count / total_count,
            TOP10: top10_count / total_count,
            TOP_ALL_BUT_ONE: toplast_count / total_count
        }


class MajorityClassifier(AttackClassifier):

    def __init__(self):
        self._clf: Dict[Tuple[int, ...], List[int]] = dict()
        self._most_freq = 0

    @property
    def name(self) -> str:
        return MAJORITY

    def fit(self, inputs: np.ndarray, labels: np.ndarray):
        """
        Fits the majority classifier which maps labels to the most
        frequent label for each level count.

        Args:
            inputs: A [B, T, D] array of input features (D) for each input sample (B) and window step (T)
            labels: A [B] array of data labels for each input sample (B)
        """
        assert len(inputs.shape) == 3, 'Must provide a 3d array of inputs'
        assert len(labels.shape) == 1, 'Must provide a 1d array of labels'
        assert inputs.shape[0] == labels.shape[0], 'Inputs and Labels are misaligned'

        label_counts: DefaultDict[int, List[int]] = defaultdict(list)
        num_labels = np.max(labels) + 1
        input_range = np.max(inputs) + 1

        for input_features, label in zip(inputs, labels):
            exit_counts = tuple(np.sum(input_features, axis=1))  # D
            label_counts[exit_counts].append(label)

        for key, count_labels in sorted(label_counts.items()):
            freq = np.bincount(count_labels, minlength=num_labels)
            most_freq = np.argsort(freq)[::-1]
            self._clf[key] = most_freq

        label_counts = np.bincount(labels, minlength=num_labels)
        self._most_freq = np.argsort(label_counts)[::-1]

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> List[int]:
        assert len(inputs.shape) == 2, 'Must provide a 2d array of input features'
        target = tuple(np.sum(inputs, axis=1))
        best_key = None
        best_diff = BIG_NUMBER

        for key in self._clf.keys():
            diff = sum((abs(k - t) for k, t in zip(key, target)))
            if diff < best_diff:
                best_diff = diff
                best_key = key

        if best_key is None:
            rankings = self._most_freq
        else:
            rankings = self._clf.get(best_key, self._most_freq)

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
            inputs: A [B, T, D] array of input features (D) for each input sample (B) and window step (T)
            labels: A [B] array of data labels for each input sample (B)
        """
        assert len(inputs.shape) == 3, 'Must provide a 3d array of inputs'
        assert len(labels.shape) == 1, 'Must provide a 1d array of labels'
        assert inputs.shape[0] == labels.shape[0], 'Inputs and Labels are misaligned'

        label_counts: DefaultDict[Tuple[int, ...], List[int]] = defaultdict(list)
        num_labels = np.max(labels) + 1

        for input_features, label in zip(inputs, labels):
            features = tuple(int(np.argmax(vector)) for vector in input_features)
            label_counts[features].append(label)

        for features, feature_labels in sorted(label_counts.items()):
            freq = np.bincount(feature_labels, minlength=num_labels)
            most_freq = np.argsort(freq)[::-1]
            self._clf[features] = most_freq

        label_counts = np.bincount(labels, minlength=num_labels)
        self._most_freq = np.argsort(label_counts)[::-1]

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> List[int]:
        assert len(inputs.shape) == 2, 'Must provide a 1d array of input features'
        features = tuple(int(np.argmax(vector)) for vector in inputs)

        best_key = None
        best_diff = BIG_NUMBER

        for key in self._clf.keys():
            diff = sum(abs(k - t) for k, t in zip(key, features))
            if diff < best_diff:
                best_key = key
                best_diff = diff

        if best_key is None:
            rankings = self._most_freq
        else:
            rankings = self._clf.get(best_key, self._most_freq)

        return rankings[0:top_k].astype(int).tolist()


class RateClassifier(AttackClassifier):

    def __init__(self):
        self._clf: np.ndarray = np.empty(0)  # Maps labels to exit rate
        self._label_freq: Dict[int, float] = dict()
        self._cutoff = 0.0

    @property
    def name(self) -> str:
        return RATE

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
        label_counter: Counter = Counter()
        num_labels = np.max(labels) + 1
        self._cutoff = 1.0 / num_labels

        for input_features, label in zip(inputs, labels):
            label_counts[label].extend(((input_features + 1) / 2).astype(int))
            label_counter[label] += 1

        self._clf = np.zeros(num_labels, dtype=float)  # [L]

        for label, counts in label_counts.items():
            elev_rate = np.average(counts)
            self._clf[label] = elev_rate

        total_count = sum(label_counter.values())
        for label in range(num_labels):
            self._label_freq[label] = label_counter.get(label, 0) / total_count

        return self._clf

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> List[int]:
        assert len(inputs.shape) == 1, 'Must provide a 1d array of input features'

        observed_elev = np.sum(((inputs + 1) / 2).astype(int))
        expected_elev = inputs.shape[0] * self._clf

        diff = np.abs(observed_elev - expected_elev)
        rankings = np.argsort(diff)

        above_rankings: List[int] = []
        below_rankings: List[int] = []

        for label in rankings:
            if self._label_freq[label] < self._cutoff:
                below_rankings.append(label)
            else:
                above_rankings.append(label)

        concat_rankings = above_rankings + below_rankings
        return concat_rankings[0:top_k]


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
