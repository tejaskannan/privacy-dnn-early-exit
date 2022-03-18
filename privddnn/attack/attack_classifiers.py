import numpy as np
from annoy import AnnoyIndex
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, DefaultDict

from privddnn.utils.constants import BIG_NUMBER


MAJORITY = 'Majority'
MOST_FREQ = 'MostFrequent'
LOGISTIC_REGRESSION_COUNT = 'LogisticRegressionCount'
LOGISTIC_REGRESSION_NGRAM = 'LogisticRegressionNgram'
DECISION_TREE = 'DecisionTree'
NGRAM = 'Ngram'
WINDOW_NGRAM = 'WindowNgram'
RATE = 'Rate'


ACCURACY = 'accuracy'
TOP2 = 'top2'
TOP5 = 'top5'
TOP10 = 'top10'
TOP_UNTIL_90 = 'topUntil90'
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

    def score(self, inputs: np.ndarray, labels: np.ndarray, num_labels: int) -> Dict[str, float]:
        """
        Evaluates the classifier on the given dataset.

        Args:
            inputs: A [B, T, D] array of input features (D) for each sample (B) and time step (T)
            labels: A [B] array of data labels.
        Returns:
            A dictionary of score metric name -> metric value
        """
        assert len(inputs.shape) == 3, 'Must provide a 2d array of inputs'
        assert len(labels.shape) == 1, 'Must provide a 1d array of labels'
        assert inputs.shape[0] == labels.shape[0], 'Inputs and Labels are misaligned'

        correct_count = 0.0
        top2_count = 0.0
        top5_count = 0.0
        top10_count = 0.0
        toplast_count = 0.0
        total_count = 0.0

        rankings_list: List[List[int]] = []

        for count, label in zip(inputs, labels):
            rankings = self.predict_rankings(count, top_k=num_labels)

            top2_count += float(label in rankings[0:2])
            top5_count += float(label in rankings[0:5])
            top10_count += float(label in rankings[0:10])
            toplast_count += float(label in rankings[0:num_labels-1])
            correct_count += float(rankings[0] == label)
            total_count += 1.0
            rankings_list.append(np.expand_dims(rankings, axis=0))

        rankings_array = np.vstack(rankings_list)  # [N, L]
        labels_array = np.expand_dims(labels, axis=-1)  # [N, 1]

        top_until_90 = num_labels
        for topk in range(1, num_labels + 1):
            topk_rankings = rankings_array[:, 0:topk]  # [N, K]
            is_in_topk = np.max(np.equal(topk_rankings, labels_array), axis=-1)  # [N]
            topk_accuracy = np.average(is_in_topk.astype(float))

            if topk_accuracy >= 0.9:
                top_until_90 = topk
                break

        return {
            ACCURACY: correct_count / total_count,
            TOP2: top2_count / total_count,
            TOP5: top5_count / total_count,
            TOP10: top10_count / total_count,
            TOP_ALL_BUT_ONE: toplast_count / total_count,
            TOP_UNTIL_90: top_until_90
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
        features_array = np.sum(inputs, axis=1).astype(int)  # [B, D]

        for input_features, label in zip(features_array, labels):
            exit_counts = tuple(input_features)  # D
            label_counts[exit_counts].append(label)

        for key, count_labels in sorted(label_counts.items()):
            freq = np.bincount(count_labels, minlength=num_labels)
            most_freq = np.argsort(freq)[::-1]
            self._clf[key] = most_freq

        label_counts = np.bincount(labels, minlength=num_labels)
        self._most_freq = np.argsort(label_counts)[::-1]

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> List[int]:
        assert len(inputs.shape) == 2, 'Must provide a 2d array of input features'
        target = tuple(np.sum(inputs, axis=0))
        best_key = None
        best_diff = BIG_NUMBER

        if target in self._clf:
            rankings = self._clf[target]
        else:
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

        if features in self._clf:
            rankings = self._clf[features]
        else:
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


class WindowNgramClassifier(AttackClassifier):

    def __init__(self, window_size: int):
        self._clf: Dict[Tuple[int, ...], np.ndarray] = dict()
        self._most_freq = 0
        self._window_size = window_size

    @property
    def name(self) -> str:
        return WINDOW_NGRAM

    @property
    def window_size(self) -> int:
        return self._window_size

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
            features_list: List[int] = []

            for idx in range(0, len(input_features) - self.window_size + 1, self.window_size):
                input_slice = input_features[idx:idx+self.window_size]  # [W, D]
                window_features = np.sum(input_slice, axis=0).astype(int)  # [D]
                features_list.extend(window_features)

            features = tuple(features_list)
            label_counts[features].append(label)

        for features, feature_labels in sorted(label_counts.items()):
            freq = np.bincount(feature_labels, minlength=num_labels)
            most_freq = np.argsort(freq)[::-1]
            self._clf[features] = most_freq

        label_counts = np.bincount(labels, minlength=num_labels)
        self._most_freq = np.argsort(label_counts)[::-1]

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> List[int]:
        assert len(inputs.shape) == 2, 'Must provide a 1d array of input features'
        label_counter: Counter = Counter()

        features_list: List[int] = []
        for idx in range(0, len(inputs) - self.window_size + 1, self.window_size):
            input_slice = inputs[idx:idx+self.window_size]  # [W, D]
            window_features = np.sum(input_slice, axis=0).astype(int)
            features_list.extend(window_features)

        features = tuple(features_list)

        if features in self._clf:
            rankings = self._clf[features]
        else:
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

    def __init__(self, mode: str):
        assert mode in ('count', 'ngram'), 'Mode must be one of `count`, `ngram`. Got: {}'.format(mode)
        self._scaler = StandardScaler()
        self._clf = None
        self._mode = mode

    def fit(self, inputs: np.ndarray, labels: np.ndarray):
        """
        Fits a classifier for the given dataset.
        """
        assert self._clf is not None, 'Subclass must set a classifier'
        input_features = self.make_input_features(inputs)
        scaled_inputs = self._scaler.fit_transform(input_features)
        self._clf.fit(scaled_inputs, labels)

    def make_input_features(self, inputs: np.ndarray) -> np.ndarray:
        assert len(inputs.shape) in (2, 3), 'Inputs array must be either 2d or 3d'

        if self._mode == 'count':
            if len(inputs.shape) == 2:
                return np.sum(inputs, axis=0)
            else:
                return np.sum(inputs, axis=1)
        elif self._mode == 'ngram':
            if len(inputs.shape) == 2:
                return inputs.reshape(-1)
            else:
                num_samples = inputs.shape[0]
                return inputs.reshape(num_samples, -1)
        else:
            raise ValueError('Unknown mode: {}'.format(self._mode))

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> List[int]:
        assert len(inputs.shape) == 2, 'Must provide a 2d array of inputs'
        assert self._clf is not None, 'Subclass must set a classifier'

        input_features = self.make_input_features(inputs)  # K
        scaled_inputs = self._scaler.transform(input_features)

        probs = self._clf.predict_proba(scaled_inputs)[0]  # [L]
        rankings = np.argsort(probs)[::-1]

        return rankings[0:top_k].astype(int).tolist()

    def score(self, inputs: np.ndarray, labels: np.ndarray, num_labels: int) -> Dict[str, float]:
        """
        Evaluates the classifier on the given dataset.

        Args:
            inputs: A [B, D] array of input features (D) for each sample (B)
            labels: A [B] array of data labels.
        Returns:
            A dictionary of score metric name -> metric value
        """
        assert len(inputs.shape) == 3, 'Must provide a 2d array of inputs'
        assert len(labels.shape) == 1, 'Must provide a 1d array of labels'
        assert inputs.shape[0] == labels.shape[0], 'Inputs and Labels are misaligned'
        assert self._clf is not None, 'Subclass must set a classifier'

        input_features = self.make_input_features(inputs)
        scaled_inputs = self._scaler.transform(input_features)
        probs = self._clf.predict_proba(scaled_inputs)  # [B, L]
        preds = np.argmax(probs, axis=-1)

        label_space = list(range(num_labels))

        # If needed, remap the lists of labels into the full space
        if probs.shape[-1] < num_labels:
            probs_list: List[np.ndarray] = []  # List of [B, 1] arrays
            num_samples = probs.shape[0]
            class_list = self._clf.classes_.astype(int).tolist()

            for label in label_space:
                if label in class_list:
                    label_idx = class_list.index(label)
                    probs_list.append(np.expand_dims(probs[:, label_idx], axis=-1))
                else:
                    probs_list.append(np.zeros(shape=(num_samples, 1)))
                
            probs = np.concatenate(probs_list, axis=-1)

        accuracy = accuracy_score(y_true=labels, y_pred=np.argmax(probs, axis=-1))
        top2 = top_k_accuracy_score(y_true=labels, y_score=probs, k=2, labels=label_space)
        top5 = top_k_accuracy_score(y_true=labels, y_score=probs, k=5, labels=label_space) if num_labels > 5 else 1.0
        top10 = top_k_accuracy_score(y_true=labels, y_score=probs, k=10, labels=label_space) if num_labels > 10 else 1.0
        top_last = top_k_accuracy_score(y_true=labels, y_score=probs, k=num_labels - 1, labels=label_space)

        top_until_90 = num_labels
        for topk in range(1, num_labels):
            top_accuracy = top_k_accuracy_score(y_true=labels, y_score=probs, k=topk, labels=label_space)
            if top_accuracy >= 0.9:
                top_until_90 = topk
                break

        return {
            ACCURACY: float(accuracy),
            TOP2: float(top2),
            TOP5: float(top5),
            TOP10: float(top10),
            TOP_ALL_BUT_ONE: float(top_last),
            TOP_UNTIL_90: int(top_until_90)
        }


class LogisticRegressionCount(SklearnClassifier):

    def __init__(self):
        super().__init__(mode='count')
        self._clf = LogisticRegression(random_state=243089, max_iter=1000)

    @property
    def name(self) -> str:
        return LOGISTIC_REGRESSION_COUNT


class LogisticRegressionNgram(SklearnClassifier):

    def __init__(self):
        super().__init__(mode='ngram')
        self._clf = LogisticRegression(random_state=243089, max_iter=1000)

    @property
    def name(self) -> str:
        return LOGISTIC_REGRESSION_NGRAM


class DecisionTreeEnsemble(SklearnClassifier):

    def __init__(self):
        super().__init__(mode='ngram')
        self._clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
                                       n_estimators=100,
                                       random_state=90423)

    @property
    def name(self) -> str:
        return DECISION_TREE
