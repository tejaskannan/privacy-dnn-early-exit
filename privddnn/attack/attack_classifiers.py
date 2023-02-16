import numpy as np
from annoy import AnnoyIndex
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score, ndcg_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, DefaultDict, Tuple

from privddnn.utils.constants import BIG_NUMBER
from privddnn.utils.metrics import compute_entropy, compute_avg_correct_rank_from_probs
from privddnn.utils.file_utils import read_pickle_gz, save_pickle_gz


MAJORITY = 'Majority'
MOST_FREQ = 'MostFrequent'
LOGISTIC_REGRESSION_COUNT = 'LogisticRegressionCount'
LOGISTIC_REGRESSION_NGRAM = 'LogisticRegressionNgram'
DECISION_TREE_COUNT = 'DecisionTreeCount'
DECISION_TREE_NGRAM = 'DecisionTreeNgram'
NGRAM = 'Ngram'
WINDOW_NGRAM = 'WindowNgram'
RATE = 'Rate'


ACCURACY = 'accuracy'
TOP2 = 'top2'
TOP5 = 'top5'
TOP10 = 'top10'
TOP_UNTIL_90 = 'topUntil90'
TOP_ALL_BUT_ONE = 'top(k-1)'
WEIGHTED_ACCURACY = 'weighted_accuracy'
AVG_CORRECT_RANK = 'correct_rank'
STD_CORRECT_RANK = 'correct_rank_std'
CONFUSION_MATRIX = 'confusion_matrix'


class AttackClassifier:

    @property
    def num_labels(self) -> int:
        return self._num_labels

    def name(self) -> str:
        raise NotImplementedError()

    def fit(self, inputs: np.ndarray, labels: np.ndarray, num_labels: int):
        raise NotImplementedError()

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> Tuple[List[int], float]:
        raise NotImplementedError()

    def predict(self, inputs: np.ndarray) -> int:
        return self.predict_rankings(inputs=inputs, top_k=1)[0][0]

    def score(self, inputs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
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
        weighted_count = 0.0

        total_count = 0.0
        weight_sum = 0.0

        rankings_list: List[List[int]] = []
        correct_ranks: List[int] = []

        for count, label in zip(inputs, labels):
            rankings, confidence = self.predict_rankings(count, top_k=self.num_labels)

            top2_count += float(label in rankings[0:2])
            top5_count += float(label in rankings[0:5])
            top10_count += float(label in rankings[0:10])
            toplast_count += float(label in rankings[0:self.num_labels-1])
            correct_count += float(rankings[0] == label)
            weighted_count += confidence * float(rankings[0] == label)

            rank = np.argmax(np.equal(rankings, label)) + 1.0
            correct_ranks.append(rank)

            total_count += 1.0
            weight_sum += confidence

            rankings_list.append(np.expand_dims(rankings, axis=0))

        rankings_array = np.vstack(rankings_list)  # [N, L]
        labels_array = np.expand_dims(labels, axis=-1)  # [N, 1]

        # Compute the confusion matrix
        confusion_mat = confusion_matrix(y_true=labels, y_pred=rankings[: 0]).astype(int).tolist()  # [L, L]

        top_until_90 = self.num_labels
        for topk in range(1, self.num_labels + 1):
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
            TOP_UNTIL_90: int(top_until_90),
            WEIGHTED_ACCURACY: weighted_count / max(weight_sum, 1.0),
            AVG_CORRECT_RANK: np.average(correct_ranks),
            STD_CORRECT_RANK: np.std(correct_ranks),
            CONFUSION_MATRIX: confusion_mat
        }


class MajorityClassifier(AttackClassifier):

    def __init__(self):
        self._clf: Dict[Tuple[int, ...], List[int]] = dict()
        self._confidence: Dict[Tuple[int, ...], float] = dict()
        self._most_freq = 0

    @property
    def name(self) -> str:
        return MAJORITY

    def fit(self, inputs: np.ndarray, labels: np.ndarray, num_labels: int):
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
        input_range = np.max(inputs) + 1
        features_array = np.sum(inputs, axis=1).astype(int)  # [B, D]
        self._num_labels = num_labels

        for input_features, label in zip(features_array, labels):
            exit_counts = tuple(input_features)  # D
            label_counts[exit_counts].append(label)

        max_dist = np.ones(shape=(self.num_labels, )).astype(float) / self.num_labels
        max_entropy = compute_entropy(max_dist, axis=-1)

        for key, count_labels in sorted(label_counts.items()):
            bincounts = np.bincount(count_labels, minlength=self.num_labels)
            freq = bincounts / np.sum(bincounts)

            most_freq = np.argsort(freq)[::-1]
            self._clf[key] = most_freq
            self._confidence[key] = 1.0 - max(compute_entropy(freq, axis=-1), 0.0) / max_entropy

        label_counts = np.bincount(labels, minlength=self.num_labels)
        self._most_freq = np.argsort(label_counts)[::-1]

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> List[int]:
        assert len(inputs.shape) == 2, 'Must provide a 2d array of input features'
        target = tuple(np.sum(inputs, axis=0))
        best_key = None
        best_diff = BIG_NUMBER

        if target in self._clf:
            rankings = self._clf[target]
            confidence = self._confidence[target]
        else:
            for key in self._clf.keys():
                diff = sum((abs(k - t) for k, t in zip(key, target)))
                if diff < best_diff:
                    best_diff = diff
                    best_key = key

            if best_key is None:
                rankings = self._most_freq
                confidence = 0.0
            else:
                rankings = self._clf.get(best_key, self._most_freq)
                confidence = self._confidence.get(best_key, 0.0)

        return rankings[0:top_k].astype(int).tolist(), confidence


class NgramClassifier(AttackClassifier):

    def __init__(self, num_neighbors: int):
        self._clf: Dict[Tuple[int, ...], np.ndarray] = dict()
        self._confidence: Dict[Tuple[int, ...], float] = dict()
        self._most_freq = 0
        self._num_neighbors = num_neighbors

    @property
    def name(self) -> str:
        return NGRAM

    @property
    def num_neighbors(self) -> int:
        return self._num_neighbors

    def fit(self, inputs: np.ndarray, labels: np.ndarray, num_labels: int):
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

        # Make the nearest neighbor index
        _, window_size, num_exits = inputs.shape
        index_features = num_exits * window_size
        self._num_labels = num_labels

        self._nn_index = AnnoyIndex(index_features, metric='euclidean')

        # Make the reverse index for predictions and add all features to the nearest
        # neighbor index
        self._pred_counters: Dict[int, Counter] = dict()
        features_num = 0

        self._reverse_index: Dict[Tuple[int, ...], int] = dict()

        for input_features, label in zip(inputs, labels):
            features = input_features.reshape(-1).astype(int).tolist()
            features_tuple = tuple(features)
            features_idx = self._reverse_index.get(features_tuple, -1)

            if features_idx == -1:
                self._pred_counters[features_num] = Counter()
                self._reverse_index[features_tuple] = features_num
                self._nn_index.add_item(features_num, features)
                features_num += 1

            features_idx = self._reverse_index[features_tuple]
            self._pred_counters[features_idx][label] += 1

        # Build the nearest neighbor index
        self._nn_index.build(32)

        label_counts = np.bincount(labels, minlength=self.num_labels)
        self._most_freq = int(np.argmax(label_counts))

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> Tuple[List[int], float]:
        assert len(inputs.shape) == 2, 'Must provide a 1d array of input features'

        features = inputs.reshape(-1).astype(int).tolist()
        neighbor_indices = self._nn_index.get_nns_by_vector(features, self.num_neighbors, include_distances=False)

        label_counter: Counter = Counter()
        for neighbor_idx in neighbor_indices:
            neighbor_counters = self._pred_counters[neighbor_idx]
            label_counter.update(neighbor_counters)

        rankings_with_counts = label_counter.most_common(top_k)
        rankings = list(map(lambda t: t[0], rankings_with_counts))

        while len(rankings) < top_k:
            rankings.append(self._most_freq)

        return rankings, 1.0


class WindowNgramClassifier(AttackClassifier):

    def __init__(self, window_size: int, num_neighbors: int):
        self._clf: Dict[Tuple[int, ...], np.ndarray] = dict()
        self._most_freq = 0
        self._window_size = window_size
        self._num_neighbors = num_neighbors

    @property
    def name(self) -> str:
        return WINDOW_NGRAM

    @property
    def num_neighbors(self) -> int:
        return self._num_neighbors

    @property
    def window_size(self) -> int:
        return self._window_size

    def fit(self, inputs: np.ndarray, labels: np.ndarray, num_labels: int):
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

        # Make the nearest neighbor index
        _, seq_size, num_exits = inputs.shape
        index_features = num_exits * self.window_size
        self._nn_indices = [AnnoyIndex(index_features, metric='euclidean') for _ in range(0, seq_size - self.window_size + 1, self.window_size)]
        self._num_labels = num_labels

        # Make the reverse index for predictions and add all features to the nearest
        # neighbor index
        self._pred_counters: DefaultDict[int, Dict[int, Counter]] = defaultdict(dict)
        feature_nums = [0 for _ in range(len(self._nn_indices))]

        self._reverse_index: DefaultDict[int, Dict[Tuple[int, ...], int]] = defaultdict(dict)

        for input_features, label in zip(inputs, labels):

            for nn_idx, window_idx in enumerate(range(0, seq_size - self.window_size + 1, self.window_size)):
                window_features = input_features[window_idx:window_idx+self.window_size]
                if len(window_features) < self.window_size:
                    continue

                features = window_features.reshape(-1).astype(int).tolist()
                features_tuple = tuple(features)
                features_idx = self._reverse_index[nn_idx].get(features_tuple, -1)

                if features_idx == -1:
                    self._pred_counters[nn_idx][feature_nums[nn_idx]] = Counter()
                    self._reverse_index[nn_idx][features_tuple] = feature_nums[nn_idx]
                    self._nn_indices[nn_idx].add_item(feature_nums[nn_idx], features)
                    feature_nums[nn_idx] += 1

                features_idx = self._reverse_index[nn_idx][features_tuple]
                self._pred_counters[nn_idx][features_idx][label] += 1

        # Build the nearest neighbor index
        for nn_index in self._nn_indices:
            nn_index.build(32)

        label_counts = np.bincount(labels, minlength=self.num_labels)
        self._most_freq = int(np.argmax(label_counts))

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> Tuple[List[int], float]:
        assert len(inputs.shape) == 2, 'Must provide a 1d array of input features'

        label_counter: Counter = Counter()
        seq_size = inputs.shape[0]

        for nn_idx, window_idx in enumerate(range(0, seq_size - self.window_size + 1, self.window_size)):
            window_features = inputs[window_idx:window_idx+self.window_size]
            if len(window_features) < self.window_size:
                continue

            features = window_features.reshape(-1).astype(int).tolist()

            neighbor_indices = self._nn_indices[nn_idx].get_nns_by_vector(features, self.num_neighbors, include_distances=False)
            for neighbor_idx in neighbor_indices:
                neighbor_counter = self._pred_counters[nn_idx][neighbor_idx]
                label_counter.update(neighbor_counter)

        rankings_with_counts = label_counter.most_common(top_k)
        rankings = list(map(lambda t: t[0], rankings_with_counts))

        while len(rankings) < top_k:
            rankings.append(self._most_freq)

        return rankings, 1.0


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

    def predict_rankings(self, inputs: np.ndarray, top_k: int) -> Tuple[List[int], float]:
        count = np.sum(inputs)
        level = int(count >= (self.window / 2.0))
        rankings = self._clf[level]
        return rankings[0:top_k].astype(int).tolist(), 1.0


class SklearnClassifier(AttackClassifier):

    def __init__(self, mode: str):
        assert mode in ('count', 'ngram'), 'Mode must be one of `count`, `ngram`. Got: {}'.format(mode)
        self._scaler = StandardScaler()
        self._clf = None
        self._mode = mode

        self._window_size = -1
        self._num_labels = -1

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def num_labels(self) -> int:
        return self._num_labels

    def fit(self, inputs: np.ndarray, labels: np.ndarray, num_labels: int):
        """
        Fits a classifier for the given dataset.
        """
        assert self._clf is not None, 'Subclass must set a classifier'
        assert len(inputs.shape) == 3, 'Must provide a 3d array of inputs'

        self._window_size = inputs.shape[1]
        self._num_labels = num_labels

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

        input_features = self.make_input_features(inputs)  # [K]
        input_features = np.expand_dims(input_features, axis=0)  # [1, K]

        scaled_inputs = self._scaler.transform(input_features)
        probs = self._clf.predict_proba(scaled_inputs)[0]  # [L]

        if probs.shape[-1] < self.num_labels:
            class_list = self._clf.classes_.astype(int).tolist()
            probs_list: List[float] = []

            for label in range(self.num_labels):
                if label in class_list:
                    label_idx = class_list.index(label)
                    probs_list.append(probs[label_idx])
                else:
                    probs_list.append(0.0)

            probs = np.vstack(probs_list).reshape(-1) 

        rankings = np.argsort(probs)[::-1]
        confidence = float(probs[0])

        return rankings[0:top_k].astype(int).tolist(), confidence

    def score(self, inputs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the classifier on the given dataset.

        Args:
            inputs: A [B, D] array of input features (D) for each sample (B)
            labels: A [B] array of data labels.
        Returns:
            A dictionary of score metric name -> metric value
        """
        assert len(inputs.shape) == 3, 'Must provide a 3d array of inputs'
        assert len(labels.shape) == 1, 'Must provide a 1d array of labels'
        assert inputs.shape[0] == labels.shape[0], 'Inputs and Labels are misaligned'
        assert self._clf is not None, 'Subclass must set a classifier'

        input_features = self.make_input_features(inputs)
        scaled_inputs = self._scaler.transform(input_features)
        probs = self._clf.predict_proba(scaled_inputs)  # [B, L]
        preds = np.argmax(probs, axis=-1)
        confidence = probs[:, 0]  # [B]

        label_space = list(range(self.num_labels))

        # If needed, remap the lists of labels into the full space
        if probs.shape[-1] < self.num_labels:
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

        preds = np.argmax(probs, axis=-1)

        accuracy = accuracy_score(y_true=labels, y_pred=preds)
        top2 = top_k_accuracy_score(y_true=labels, y_score=probs, k=2, labels=label_space)
        top5 = top_k_accuracy_score(y_true=labels, y_score=probs, k=5, labels=label_space) if self.num_labels > 5 else 1.0
        top10 = top_k_accuracy_score(y_true=labels, y_score=probs, k=10, labels=label_space) if self.num_labels > 10 else 1.0
        top_last = top_k_accuracy_score(y_true=labels, y_score=probs, k=self.num_labels - 1, labels=label_space)
        weighted_accuracy = accuracy_score(y_true=labels, y_pred=preds, sample_weight=confidence)
        confusion_mat = confusion_matrix(y_true=labels, y_pred=preds).astype(int).tolist()  # [L, L]

        top_until_90 = self.num_labels
        for topk in range(1, self.num_labels):
            top_accuracy = top_k_accuracy_score(y_true=labels, y_score=probs, k=topk, labels=label_space)
            if top_accuracy >= 0.9:
                top_until_90 = topk
                break

        correct_rank = compute_avg_correct_rank_from_probs(probs=probs, labels=labels)

        return {
            ACCURACY: float(accuracy),
            TOP2: float(top2),
            TOP5: float(top5),
            TOP10: float(top10),
            TOP_ALL_BUT_ONE: float(top_last),
            TOP_UNTIL_90: int(top_until_90),
            WEIGHTED_ACCURACY: float(weighted_accuracy),
            AVG_CORRECT_RANK: np.average(correct_rank),
            STD_CORRECT_RANK: np.std(correct_rank),
            CONFUSION_MATRIX: confusion_mat
        }

    def save(self, path: str):
        model = {
            'mode': self._mode,
            'scaler': self._scaler,
            'clf': self._clf,
            'window_size': self.window_size,
            'num_labels': self.num_labels
        }

        save_pickle_gz(model, path)

    @classmethod
    def restore(cls, path: str):
        model_dict = read_pickle_gz(path)

        model = cls(mode=model_dict['mode'])
        model._scaler = model_dict['scaler']
        model._clf = model_dict['clf']

        return model


class LogisticRegressionCount(SklearnClassifier):

    def __init__(self):
        super().__init__(mode='count')
        self._clf = LogisticRegression(random_state=243089, max_iter=1000)

    @property
    def name(self) -> str:
        return LOGISTIC_REGRESSION_COUNT

    @classmethod
    def restore(cls, path: str, window_size: int, num_labels: int):
        model_dict = read_pickle_gz(path)

        model = cls()
        model._scaler = model_dict['scaler']
        model._clf = model_dict['clf']
        model._window_size = model_dict.get('window_size', window_size)
        model._num_labels = model_dict.get('num_labels', num_labels)

        return model


class LogisticRegressionNgram(SklearnClassifier):

    def __init__(self):
        super().__init__(mode='ngram')
        self._clf = LogisticRegression(random_state=243089, max_iter=1000)

    @property
    def name(self) -> str:
        return LOGISTIC_REGRESSION_NGRAM


class DecisionTreeEnsembleCount(SklearnClassifier):

    def __init__(self):
        super().__init__(mode='count')
        self._clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
                                       n_estimators=100,
                                       random_state=90423)

    @property
    def name(self) -> str:
        return DECISION_TREE_COUNT


class DecisionTreeEnsembleNgram(SklearnClassifier):

    def __init__(self):
        super().__init__(mode='ngram')
        self._clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
                                       n_estimators=100,
                                       random_state=90423)

    @property
    def name(self) -> str:
        return DECISION_TREE_NGRAM
