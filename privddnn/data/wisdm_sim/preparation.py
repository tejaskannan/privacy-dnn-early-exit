import numpy as np
import h5py
from collections import namedtuple, deque, Counter
from typing import Iterable, Tuple, List


PATH = '/local/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
WINDOW_SIZE = 50
STEP_SIZE = 5
WALKING_TRAIN_FRAC = 0.2411
WALKING_VAL_FRAC = 0.3079
JOGGING_TRAIN_FRAC = 0.3098
JOGGING_VAL_FRAC = 0.3621

TRAIN_USERS = [14, 1, 22, 33, 10, 9, 30, 29, 28, 5, 17, 35, 6, 24, 26, 19, 11, 7, 36, 2, 25, 4, 27, 18, 20]
VAL_USERS = [12, 13, 8]
TEST_USERS = [21, 34, 23, 3, 31, 16, 32, 15]

LABEL_MAP = {
    'Walking': 0,
    'Jogging': 1,
    'Sitting': 2,
    'Standing': 2
}

DataSample = namedtuple('DataSample', ['inputs', 'label', 'user'])


def get_majority(label_window: deque) -> int:
    label_counter: Counter = Counter()
    for label in label_window:
        label_counter[label] += 1

    return label_counter.most_common(1)[0][0]


def make_data_windows(path: str) -> Iterable[Tuple[np.ndarray, int, int]]:
    step_counter = STEP_SIZE
    data_window = deque()
    label_window = deque()
    prev_user_id = -1

    rand = np.random.RandomState(5029834)

    with open(path, 'r') as fin:
        for idx, line in enumerate(fin):
            tokens = [t.strip() for t in line.split(',') if len(t.strip()) > 0]
            if len(tokens) != 6:
                continue

            user_id = int(tokens[0])
            if tokens[1] not in LABEL_MAP:
                continue

            label = LABEL_MAP[tokens[1]]
            input_features = np.expand_dims([float(x.replace(';', '')) for x in tokens[3:6]], axis=0)

            if (prev_user_id != user_id) and (idx > 0):
                step_counter = STEP_SIZE
                data_window = deque()
                label_window = deque()

            data_window.append(input_features)
            label_window.append(label)
            prev_user_id = user_id

            while len(data_window) > WINDOW_SIZE:
                data_window.popleft()

            while len(label_window) > WINDOW_SIZE:
                label_window.popleft()

            if (len(data_window) == WINDOW_SIZE) and (step_counter == 0):
                input_window = np.expand_dims(np.vstack(data_window), axis=0)
                label = get_majority(label_window)

                yield input_window, label, user_id

                step_counter = STEP_SIZE
            else:
                step_counter = max(step_counter - 1, 0)


def write_data_fold(inputs: np.ndarray, labels: np.ndarray, output_path: str):
    print(inputs.shape)
    print(labels.shape)

    with h5py.File(output_path, 'w') as fout:
        input_ds = fout.create_dataset('inputs', shape=inputs.shape, dtype='f')
        input_ds.write_direct(inputs)

        label_ds = fout.create_dataset('labels', shape=labels.shape, dtype='i')
        label_ds.write_direct(labels)


def print_label_rates(label_counter: Counter):
    total_count = sum(label_counter.values())
    for label, count in sorted(label_counter.items()):
        print('\t{} -> {} ({:.4f})'.format(label, count, count / total_count))


if __name__ == '__main__':

    train_inputs: List[np.ndarray] = []
    train_labels: List[int] = []

    val_inputs: List[np.ndarray] = []
    val_labels: List[int] = []

    test_inputs: List[np.ndarray] = []
    test_labels: List[int] = []

    train_label_counter: Counter = Counter()
    val_label_counter: Counter = Counter()
    test_label_counter: Counter = Counter()

    rand = np.random.RandomState(4237892349)

    for idx, (input_features, label, user_id) in enumerate(make_data_windows(path=PATH)):
        if user_id in TEST_USERS:
            r = rand.uniform()
            if r < 0.34:
                test_inputs.append(input_features)
                test_labels.append(label)
                test_label_counter[label] += 1
        elif user_id in VAL_USERS:
            r = rand.uniform()
            if label == 0:
                should_add = (r < WALKING_VAL_FRAC)
            elif label == 1:
                should_add = (r < JOGGING_VAL_FRAC)
            else:
                should_add = True

            if should_add:
                val_inputs.append(input_features)
                val_labels.append(label)
                val_label_counter[label] += 1
        elif user_id in TRAIN_USERS:
            r = rand.uniform()
            if label == 0:
                should_add = (r < WALKING_TRAIN_FRAC)
            elif label == 1:
                should_add = (r < JOGGING_TRAIN_FRAC)
            else:
                should_add = True

            if should_add:
                train_inputs.append(input_features)
                train_labels.append(label)
                train_label_counter[label] += 1
        else:
            raise ValueError('Unknown user id: {}'.format(user_id))

    print('Train: {}'.format(len(train_inputs)))
    print_label_rates(train_label_counter)

    print('Val: {}'.format(len(val_inputs)))
    print_label_rates(val_label_counter)

    print('Test: {}'.format(len(test_inputs)))
    print_label_rates(test_label_counter)

    write_data_fold(inputs=np.vstack(train_inputs),
                    labels=np.vstack(train_labels).reshape(-1),
                    output_path='train.h5')

    write_data_fold(inputs=np.vstack(val_inputs),
                    labels=np.vstack(val_labels).reshape(-1),
                    output_path='val.h5')

    write_data_fold(inputs=np.vstack(test_inputs),
                    labels=np.vstack(test_labels).reshape(-1),
                    output_path='test.h5')
