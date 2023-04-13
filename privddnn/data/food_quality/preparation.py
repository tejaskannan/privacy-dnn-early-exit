import os.path
import csv
import h5py
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, List


N_COMPONENTS = 16

LABEL_MAP = {
    'strawberry': 0,
    'non-strawberry': 1
}


def get_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    labels_list: List[int] = []
    data_rows: List[List[float]] = []

    with open(path, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')

        for line_idx, line in enumerate(reader):
            if line_idx == 0:
                labels_list = list(map(lambda s: LABEL_MAP[s.lower()], line[1:]))
            else:
                row = list(map(lambda s: float(s), line[1:]))
                data_rows.append(np.expand_dims(row, axis=0))

    # Transpose the data rows
    input_features = np.vstack(data_rows)
    input_features = np.transpose(input_features)

    labels = np.vstack(labels_list).reshape(-1).astype(int)
    return input_features, labels


def upsample_inputs(inputs: np.ndarray, labels: np.ndarray, rand: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    avg_values = np.mean(inputs, axis=0, keepdims=True)  # [D]
    noise_scales = avg_values * 0.1
    noise = rand.normal(loc=0.0, scale=noise_scales, size=inputs.shape)

    noisy_inputs = inputs + noise
    full_inputs = np.concatenate([inputs, noisy_inputs], axis=0)
    full_labels = np.concatenate([labels, labels], axis=0)
    return full_inputs, full_labels


def save_dataset(inputs: np.ndarray, labels: np.ndarray, path: str):
    with h5py.File(path, 'w') as fout:
        input_ds = fout.create_dataset('inputs', shape=inputs.shape, dtype='f')
        input_ds.write_direct(inputs)

        label_ds = fout.create_dataset('labels', shape=labels.shape, dtype='i')
        label_ds.write_direct(labels)


if __name__ == '__main__':
    path = '/local/food_quality/MIR_Fruit_purees.csv'
    output_path = './'
    inputs, labels = get_dataset(path)

    rand = np.random.RandomState(seed=7409)
    train_split = int(inputs.shape[0] * 0.7)
    val_split = train_split + int(inputs.shape[0] * 0.15)

    sample_indices = np.arange(inputs.shape[0])
    rand.shuffle(sample_indices)

    train_indices = sample_indices[0:train_split]
    val_indices = sample_indices[train_split:val_split]
    test_indices = sample_indices[val_split:]

    train_inputs, train_labels = inputs[train_indices], labels[train_indices]
    val_inputs, val_labels = inputs[val_indices], labels[val_indices]
    test_inputs, test_labels = inputs[test_indices], labels[test_indices]

    # Apply PCA to reduce the number of features
    pca = PCA(N_COMPONENTS)
    pca.fit(train_inputs)

    train_inputs = pca.transform(train_inputs)
    val_inputs = pca.transform(val_inputs)
    test_inputs = pca.transform(test_inputs)

    #train_inputs, train_labels = upsample_inputs(inputs=train_inputs, labels=train_labels, rand=rand)
    #val_inputs, val_labels = upsample_inputs(inputs=val_inputs, labels=val_labels, rand=rand)

    train_label_counts = np.bincount(train_labels, minlength=2) / train_labels.shape[0]
    val_label_counts = np.bincount(val_labels, minlength=2) / val_labels.shape[0]
    test_label_counts = np.bincount(test_labels, minlength=2) / test_labels.shape[0]
    print(train_label_counts)
    print(val_label_counts)
    print(test_label_counts)

    save_dataset(inputs=train_inputs, labels=train_labels, path=os.path.join(output_path, 'train.h5'))
    save_dataset(inputs=val_inputs, labels=val_labels, path=os.path.join(output_path, 'val.h5'))
    save_dataset(inputs=test_inputs, labels=test_labels, path=os.path.join(output_path, 'test.h5'))
