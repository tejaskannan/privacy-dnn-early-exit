import numpy as np
import h5py
from collections import namedtuple, deque, Counter
from typing import Iterable, Tuple, List


PATH = '/local/WISDM_at_latest/home/share/data/public_sets/WISDM_at_v2.0/WISDM_at_v2.0_raw.txt'
WINDOW_SIZE = 50
STEP_SIZE = 25

TRAIN_USERS = [1761, 720, 633, 622, 1603, 624, 594, 668, 597, 1649, 998, 681, 618, 1269, 561, 1778, 588, 1511, 669, 684, 706, 1759, 1736, 582, 721, 1554, 661, 1104, 616, 645, 663, 658, 730, 692, 1742, 591, 635, 579, 1715, 676, 1117, 1783, 1477, 632, 623, 678, 1676, 688, 634, 615, 592, 589, 636, 722, 715, 627, 648, 685, 641, 584, 1320, 712, 709, 686, 666, 1480, 671, 1733, 646, 660, 1797, 717, 711, 664, 719, 612, 674, 662, 1767, 1794, 1719, 1757, 710, 1727, 617, 723, 603, 605, 705, 725, 1763, 1501, 1277, 1793, 562, 670, 1702, 675, 628, 614, 659, 598, 925, 613, 1753, 682, 679, 652, 621, 653, 1064, 1679, 600, 724, 625, 1743, 704, 693, 637, 1765, 1247, 650]
VAL_USERS = [1774, 595, 695, 583, 1707, 683, 586, 1559, 565, 563, 707, 657, 1745, 1512, 1802, 638, 1253, 654, 611, 596, 665, 580, 587, 1274, 672, 656, 1750, 729, 581, 647, 726, 1238, 649, 651, 1696, 687, 1319, 610, 1799, 1280, 1768, 1724, 601, 1491, 702, 718, 639, 655, 727, 1518, 568, 689, 602, 573]
TEST_USERS = [716, 606, 630, 585, 1769, 608, 1100, 593, 697, 714, 713, 194, 667, 696, 607, 1656, 1531, 604, 640, 728, 1638, 567, 590, 1713, 1758, 1205, 694, 626, 691, 631, 642, 703, 599, 577, 1703, 690, 609, 619, 1647, 1246, 1775, 1683, 708, 578]


WALKING_TRAIN_FRAC = 0.4233
WALKING_VAL_FRAC = 0.1458
IDLE_TRAIN_FRAC = 0.5811
IDLE_VAL_FRAC = 0.1815


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

    rand = np.random.RandomState(23490)

    for idx, (input_features, label, user_id) in enumerate(make_data_windows(path=PATH)):
        if user_id in TEST_USERS:
            test_inputs.append(input_features)
            test_labels.append(label)
            test_label_counter[label] += 1
        elif user_id in VAL_USERS:
            should_use = False

            if label == 0:
                r = rand.uniform()
                if r < WALKING_VAL_FRAC:
                    should_use = True
            elif label == 2:
                r = rand.uniform()
                if r < IDLE_VAL_FRAC:
                    should_use = True
            else:
                should_use = True

            if should_use:
                val_inputs.append(input_features)
                val_labels.append(label)
                val_label_counter[label] += 1
        elif user_id in TRAIN_USERS:
            should_use = False

            if label == 0:
                r = rand.uniform()
                if r < WALKING_TRAIN_FRAC:
                    should_use = True
            elif label == 2:
                r = rand.uniform()
                if r < IDLE_TRAIN_FRAC:
                    should_use = True
            else:
                should_use = True

            if should_use:
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
