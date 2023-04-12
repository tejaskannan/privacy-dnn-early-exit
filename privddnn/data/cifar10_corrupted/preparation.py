import tensorflow as tf2
import h5py
import os.path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

from privddnn.dataset.dataset import get_split_indices


def blur_image(img: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(img)
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))

    return np.asarray(blurred)


def save_fold(inputs: np.ndarray, labels: np.ndarray, path: str):
    with h5py.File(path, 'w') as fout:
        input_ds = fout.create_dataset('inputs', shape=inputs.shape, dtype='f')
        input_ds.write_direct(inputs)

        label_ds = fout.create_dataset('labels', shape=labels.shape, dtype='i')
        label_ds.write_direct(labels)


def process_train_val(raw_inputs: np.ndarray, raw_labels: np.ndarray, output_dir: str):
    inputs_list: List[np.ndarray] = []
    labels_list: List[int] = []

    for img, label in zip(raw_inputs, raw_labels):
        noisy_image = blur_image(img)

        inputs_list.append(np.expand_dims(noisy_image, axis=0))
        labels_list.append(label)

    inputs = np.vstack(inputs_list)
    labels = np.vstack(labels_list).reshape(-1)

    train_idx, val_idx = get_split_indices(num_samples=len(inputs), frac=0.8, is_noisy=True)

    train_inputs, train_labels = inputs[train_idx], labels[train_idx]
    val_inputs, val_labels = inputs[val_idx], labels[val_idx]

    save_fold(inputs=train_inputs, labels=train_labels, path=os.path.join(output_dir, 'train.h5'))
    save_fold(inputs=val_inputs, labels=val_labels, path=os.path.join(output_dir, 'val.h5'))


def process_test(raw_inputs: np.ndarray, raw_labels: np.ndarray, output_dir: str):
    inputs_list: List[np.ndarray] = []
    labels_list: List[int] = []

    for img, label in zip(raw_inputs, raw_labels):
        noisy_image = blur_image(img)

        inputs_list.append(np.expand_dims(noisy_image, axis=0))
        labels_list.append(label)

    inputs = np.vstack(inputs_list)
    labels = np.vstack(labels_list).reshape(-1)

    save_fold(inputs=inputs, labels=labels, path=os.path.join(output_dir, 'test.h5'))


if __name__ == '__main__':
    tf_dataset = tf2.keras.datasets.cifar10
    (X_train, y_train), (X_test, y_test) = tf_dataset.load_data()

    process_train_val(raw_inputs=X_train, raw_labels=y_train, output_dir='/local/cifar10_corrupted')
    process_test(raw_inputs=X_test, raw_labels=y_test, output_dir='/local/cifar10_corrupted')
