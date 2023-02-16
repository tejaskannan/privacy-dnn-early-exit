import numpy as np
import h5py
import os.path
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import wave

from privddnn.utils.file_utils import make_dir
from preparation import compute_spectrogram, UNK_LABEL, AUDIO_SIZE


SNR = 100.0  # The desired signal-to-noise ratio


def get_noise_scale(audio: np.ndarray) -> float:
    mean_squared = np.mean(audio * audio)
    return float(np.sqrt(mean_squared / (np.power(10.0, (SNR / 10.0)))))


def process_fold(fold: str, output_dir: str, seed: int):
    dataset = tfds.load('speech_commands', data_dir='/local/speech', split=fold)
    rand = np.random.RandomState(seed=seed)

    inputs_list: List[np.ndarray] = []
    labels_list: List[int] = []

    background_sounds: List[np.ndarray] = []
    num_bg = 0

    for example in dataset:
        audio = example['audio'].numpy()
        label = example['label'].numpy()

        if (label != UNK_LABEL) and (audio.shape[0] == AUDIO_SIZE):
            num_bg += 1 

    for example in dataset:
        audio = example['audio'].numpy()
        label = example['label'].numpy()

        if (label == UNK_LABEL) and (audio.shape[0] == AUDIO_SIZE):
            background_sounds.append(audio)

        if len(background_sounds) >= num_bg:
            break

    background_idx = 0

    for example in dataset:
        audio = example['audio'].numpy()
        label = example['label'].numpy()

        if (label != UNK_LABEL) and (audio.shape[0] == AUDIO_SIZE):
            background_sound = background_sounds[background_idx]
            audio = audio + (background_sound / 4)
            background_idx = (background_idx + 1) % len(background_sounds)

            spectrogram = compute_spectrogram(audio)
            inputs_list.append(np.expand_dims(spectrogram, axis=0))
            labels_list.append(label)

            #with wave.open('{}.wav'.format(fold), 'wb') as fout:
            #    fout.setnchannels(1)
            #    fout.setframerate(44100)
            #    fout.setsampwidth(2)
            #    fout.writeframes(audio)

            #break

    inputs = np.vstack(inputs_list)
    labels = np.vstack(labels_list).reshape(-1).astype(int)

    print('Collected {}. Dimensions: {}'.format(fold, inputs.shape))

    with h5py.File(os.path.join(output_dir, '{}.h5'.format(fold)), 'w') as fout:
        input_ds = fout.create_dataset('inputs', shape=inputs.shape, dtype='f')
        input_ds.write_direct(inputs)

        label_ds = fout.create_dataset('labels', shape=labels.shape, dtype='i')
        label_ds.write_direct(labels)


if __name__ == '__main__':
    seeds = [428039, 19235, 23460]

    output_dir = '/local/speech_white_noise'
    make_dir(output_dir)

    for fold, seed in zip(['train', 'validation', 'test'], seeds):
        process_fold(fold, output_dir=output_dir, seed=seed)

