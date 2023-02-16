import os.path
import h5py
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from privddnn.utils.constants import SMALL_NUMBER


TRAIN_FRAC = 0.8
AUDIO_LENGTH = 4000


def compute_spectrogram(audio: np.ndarray):
    assert len(audio.shape) == 1, 'Must provide a 1d array of audio signals.'
    _, _, Sxx = spectrogram(x=audio, fs=44100, nfft=256)
    return -1 * np.log10(Sxx + SMALL_NUMBER).T


def write_dataset(inputs: np.ndarray, labels: np.ndarray, path: str):
    with h5py.File(path, 'w') as fout:
        input_ds = fout.create_dataset('inputs', shape=inputs.shape, dtype='f')
        input_ds.write_direct(inputs)

        label_ds = fout.create_dataset('labels', shape=labels.shape, dtype='i')
        label_ds.write_direct(labels)



def process(output_dir: str):
    dataset = tfds.load('spoken_digit', data_dir='/local/spoken_digit', split='train')
    rand = np.random.RandomState(seed=8954)

    train_inputs_list: List[np.ndarray] = []
    train_labels_list: List[int] = []

    val_inputs_list: List[np.ndarray] = []
    val_labels_list: List[int] = []

    for example in dataset:
        audio = example['audio'].numpy()
        label = example['label'].numpy()

        if len(audio) >= AUDIO_LENGTH:
            audio = audio[0:AUDIO_LENGTH]
        else:
            audio = np.pad(audio, pad_width=(0, AUDIO_LENGTH - len(audio)), mode='constant', constant_values=0.0)

        input_features = compute_spectrogram(audio)
        input_features = np.expand_dims(input_features, axis=0)

        if rand.uniform() < TRAIN_FRAC:
            train_inputs_list.append(input_features)
            train_labels_list.append(label)
        else:
            val_inputs_list.append(input_features)
            val_labels_list.append(label)

    train_inputs = np.vstack(train_inputs_list)
    train_labels = np.vstack(train_labels_list).reshape(-1)

    val_inputs = np.vstack(val_inputs_list)
    val_labels = np.vstack(val_labels_list).reshape(-1)

    write_dataset(inputs=train_inputs, labels=train_labels, path=os.path.join(output_dir, 'train.h5'))
    write_dataset(inputs=val_inputs, labels=val_labels, path=os.path.join(output_dir, 'val.h5'))
    write_dataset(inputs=val_inputs, labels=val_labels, path=os.path.join(output_dir, 'test.h5'))


if __name__ == '__main__':
    process(output_dir='/local/spoken_digit')
