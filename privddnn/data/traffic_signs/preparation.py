import os
import h5py
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import Counter
from PIL import Image, ImageEnhance
from typing import List

from privddnn.utils.file_utils import iterate_dir


MODULUS = 2**16


def get_fold(key: str, train_frac: float, val_frac: float) -> str:
    train_split = int(train_frac * MODULUS)
    val_split = train_split + int(val_frac * MODULUS)

    h = hashlib.md5()
    h.update(key.encode())
    digest = h.hexdigest()

    hash_result = int(digest, 16) % MODULUS

    if hash_result < train_split:
        return 'train'
    elif hash_result < val_split:
        return 'val'
    else:
        return 'test'


def process_images(input_folder: str, output_folder: str, train_frac: float, val_frac: float):
    label_counts: Counter = Counter()
    total_count = 0

    input_lists = {
        'train': [],
        'val': [],
        'test': []
    }

    label_lists = {
        'train': [],
        'val': [],
        'test': []
    }

    label_counters = {
        'train': Counter(),
        'val': Counter(),
        'test': Counter()
    }

    for label_path in iterate_dir(input_folder):
        path_tokens = label_path.split(os.sep)
        label = int(path_tokens[-1] if len(path_tokens[-1]) > 0 else path_tokens[-2])

        for path in iterate_dir(label_path):
            if not path.endswith('ppm'):
                continue

            path_tokens = path.split(os.sep)
            file_name = path_tokens[-1]
            sign_idx = file_name.split('_')[0]
            key = '{}_{}'.format(label, sign_idx)
            fold = get_fold(key=key, train_frac=train_frac, val_frac=val_frac)

            img = Image.open(path).convert('L')
            img = ImageEnhance.Contrast(img).enhance(1.5)

            resized = img.resize(size=(32, 32), resample=Image.BILINEAR)
            total_count += 1

            input_lists[fold].append(np.expand_dims(np.array(resized), axis=0))
            label_lists[fold].append(label)
            label_counters[fold][label] += 1

            if fold == 'train':
                flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
                input_lists[fold].append(np.expand_dims(np.array(flipped), axis=0))
                label_lists[fold].append(label)

            if total_count % 1000 == 0:
                print('Completed {} samples.'.format(total_count), end='\r')

    print()
    print(total_count)

    for fold in input_lists.keys():
        inputs = np.expand_dims(np.vstack(input_lists[fold]), axis=-1)
        labels = np.vstack(label_lists[fold]).reshape(-1)
        output_file = os.path.join(output_folder, '{}.h5'.format(fold))

        print(output_file)
        print(inputs.shape)
        print(label_counters[fold])

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

    process_images(input_folder=args.train_folder,
                   output_folder=args.output_folder,
                   train_frac=0.5,
                   val_frac=0.25)
