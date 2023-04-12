import numpy as np
import h5py
import os.path
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import wave
from typing import List, Any, Tuple

from privddnn.utils.file_utils import make_dir
from preparation import compute_spectrogram, UNK_LABEL, AUDIO_SIZE


TRAIN_FRAC = 0.7
SNR = 50.0


def get_noise_stddev(audio: np.ndarray) -> float:
    avg_value = np.average(audio)
    mean_square = np.square(np.std(audio)) + avg_value
    return np.sqrt(mean_square / SNR)


def create_features(raw_inputs: List[np.ndarray], background_noise: List[np.ndarray]) -> np.ndarray:
    idx = 0

    input_features_list: List[np.ndarray] = []

    for audio_idx, audio_input in enumerate(raw_inputs):
        background = (background_noise[idx] / 8)
        audio = audio_input + background

        spectrogram = compute_spectrogram(audio_input)
        input_features_list.append(np.expand_dims(spectrogram, axis=0))

        fig, ax = plt.subplots()
        ax.imshow(spectrogram, cmap='gray_r')
        plt.show()

        idx = (idx + 1) % len(background_indices)

        if ((audio_idx + 1) % 100):
            print('Completed {} / {} features.'.format(audio_idx, len(raw_inputs)), end='\r')

    print()
    return np.vstack(input_features_list)


def write_fold(input_features: np.ndarray, labels: np.ndarray, path: str):
    with h5py.File(path, 'w') as fout:
        input_ds = fout.create_dataset('inputs', shape=input_features.shape, dtype='f')
        input_ds.write_direct(input_features)

        label_ds = fout.create_dataset('labels', shape=labels.shape, dtype='i')
        label_ds.write_direct(labels)


def make_noisy_fold(dataset: Any, train_input_list: List[np.ndarray], train_label_list: List[int], val_input_list: List[np.ndarray], val_label_list: List[int], rand: np.random.RandomState):
    num_inputs = 0
    for example in dataset:
        audio = example['audio'].numpy()
        label = example['label'].numpy()

        num_inputs += int((label != UNK_LABEL) and (audio.shape[0] == AUDIO_SIZE))

    inputs_list: List[np.ndarray] = []
    labels_list: List[int] = []
    bg_list: List[np.ndarray] = []

    for idx, example in enumerate(dataset):
        audio = example['audio'].numpy()
        label = example['label'].numpy()

        if audio.shape[0] != AUDIO_SIZE:
            continue

        if (label != UNK_LABEL):
            noise = rand.normal(loc=0.0, scale=get_noise_stddev(audio), size=audio.shape)
            audio = audio.astype(float) + noise

            features = compute_spectrogram(audio)

            if (rand.uniform() < TRAIN_FRAC):
                train_input_list.append(np.expand_dims(features, axis=0))
                train_label_list.append(label)
            else:
                val_input_list.append(np.expand_dims(features, axis=0))
                val_label_list.append(label)

        #if label == UNK_LABEL:
        #    if (len(bg_list) < num_inputs):
        #        bg_list.append(audio)
        #else:
        #    inputs_list.append(audio)
        #    labels_list.append(label)

        if (idx + 1) % 100 == 0:
            print('Completed {} samples'.format(idx + 1), end='\r')


def write_train_val_data(rand: np.random.RandomState, output_dir: str):
    train_dataset = tfds.load('speech_commands', data_dir='/local/speech', split='train')
    val_dataset = tfds.load('speech_commands', data_dir='/local/speech', split='validation')

    train_feature_list: List[np.ndarray] = []
    val_feature_list: List[np.ndarray] = []
    train_label_list: List[int] = []
    val_label_list: List[int] = []

    make_noisy_fold(train_dataset, train_feature_list, train_label_list, val_feature_list, val_label_list, rand=rand)
    make_noisy_fold(val_dataset, train_feature_list, train_label_list, val_feature_list, val_label_list, rand=rand)

    train_inputs = np.vstack(train_feature_list)
    train_labels = np.vstack(train_label_list).reshape(-1)

    val_inputs = np.vstack(val_feature_list)
    val_labels = np.vstack(val_label_list).reshape(-1)

    write_fold(train_inputs, train_labels, path=os.path.join(output_dir, 'train.h5'))
    write_fold(val_inputs, val_labels, path=os.path.join(output_dir, 'val.h5'))


def write_test_data(output_dir: str):
    test_dataset = tfds.load('speech_commands', data_dir='/local/speech', split='test')

    inputs_list: List[np.ndarray] = []
    labels_list: List[int] = []

    for example in test_dataset:
        audio = example['audio'].numpy()
        label = example['label'].numpy()

        if (label != UNK_LABEL) and (audio.shape[0] == AUDIO_SIZE):
            inputs_list.append(np.expand_dims(compute_spectrogram(audio), axis=0))
            labels_list.append(label)

    inputs = np.vstack(inputs_list)
    labels = np.vstack(labels_list)

    write_fold(inputs, labels, path=os.path.join(output_dir, 'test.h5'))


if __name__ == '__main__':
    output_dir = '/local/speech_background'
    make_dir(output_dir)

    rand = np.random.RandomState(3910)
    write_train_val_data(rand=rand, output_dir=output_dir)
    write_test_data(output_dir=output_dir)
