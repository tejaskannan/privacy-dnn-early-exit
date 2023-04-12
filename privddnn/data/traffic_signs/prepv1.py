import os
import h5py
import numpy as np
import hashlib
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import Counter
from PIL import Image, ImageEnhance
from typing import List

from privddnn.utils.file_utils import iterate_dir


def process_images(input_folder: str, output_folder: str, fold: str):
    label_counts: Counter = Counter()
    total_count = 0

    input_list: List[np.ndarray] = []
    label_list: List[int] = []

    with open(os.path.join(input_folder, '{}.p'.format(fold)), 'rb') as fin:
        dataset = pickle.load(fin)

    for features, label in zip(dataset['features'], dataset['labels']):
        img = Image.fromarray(features, 'RGB').convert('L')
        img = ImageEnhance.Contrast(img).enhance(1.5)

        #plt.imshow(img, cmap='gray')
        #plt.show()

        total_count += 1

        input_list.append(np.expand_dims(np.array(img), axis=0))
        label_list.append(label)
        label_counts[label] += 1

        #if fold == 'train':
        #    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        #    input_list.append(np.expand_dims(np.array(flipped), axis=0))
        #    label_list.append(label)

        if total_count % 1000 == 0:
            print('Completed {} samples.'.format(total_count), end='\r')

    print()
    print(total_count)

    inputs = np.expand_dims(np.vstack(input_list), axis=-1)
    labels = np.vstack(label_list).reshape(-1)
    output_file = os.path.join(output_folder, '{}.h5'.format(fold))

    print(output_file)
    print(inputs.shape)
    print(label_counts)

    with h5py.File(output_file, 'w') as fout:
        input_ds = fout.create_dataset('inputs', shape=inputs.shape, dtype='f')
        input_ds.write_direct(inputs)

        label_ds = fout.create_dataset('labels', shape=labels.shape, dtype='i')
        label_ds.write_direct(labels)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    args = parser.parse_args()

    for fold in ['train', 'val', 'test']:
        process_images(input_folder=args.train_folder,
                       output_folder=args.output_folder,
                       fold=fold)
