import os.path
import h5py
import numpy as np
import tensorflow_datasets as tfds
from scipy.signal import spectrogram

from privddnn.utils.constants import SMALL_NUMBER


UNK_LABEL = 11
AUDIO_SIZE = 16000


def compute_spectrogram(audio: np.ndarray):
    assert len(audio.shape) == 1, 'Must provide a 1d array of audio signals.'
    _, _, Sxx = spectrogram(x=audio, fs=44100, nfft=256)
    return -1 * np.log10(Sxx + SMALL_NUMBER).T


def process_fold(fold: str, output_dir: str):
    dataset = tfds.load('speech_commands', data_dir='/local/speech', split=fold)

    inputs_list: List[np.ndarray] = []
    labels_list: List[int] = []

    for example in dataset:
        audio = example['audio'].numpy()
        label = example['label'].numpy()

        if (label != UNK_LABEL) and (audio.shape[0] == AUDIO_SIZE):
            input_features = compute_spectrogram(audio=audio)
            inputs_list.append(np.expand_dims(input_features, axis=0))
            labels_list.append(label)

    inputs = np.vstack(inputs_list)
    labels = np.vstack(labels_list).reshape(-1)

    print('Collected {}. Dimensions: {}'.format(fold, inputs.shape))

    with h5py.File(os.path.join(output_dir, '{}.h5'.format(fold)), 'w') as fout:
        input_ds = fout.create_dataset('inputs', shape=inputs.shape, dtype='f')
        input_ds.write_direct(inputs)

        label_ds = fout.create_dataset('labels', shape=labels.shape, dtype='i')
        label_ds.write_direct(labels)


if __name__ == '__main__':
    for fold in ['train', 'validation', 'test']:
        process_fold(fold, output_dir='/local/speech')
