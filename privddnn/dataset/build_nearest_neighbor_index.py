import numpy as np
from annoy import AnnoyIndex
from argparse import ArgumentParser

from privddnn.dataset.dataset import Dataset
from privddnn.utils.file_utils import save_pickle_gz


def create_index(inputs: np.ndarray, path: str):
    num_samples = inputs.shape[0]
    num_features = inputs.shape[-1]
    flattened = inputs.reshape(num_samples, -1)

    index = AnnoyIndex(num_features, 'euclidean')
    for idx, features in enumerate(inputs):
        index.add_item(idx, features)

    index.build(100)
    index.save(path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    args = parser.parse_args()

    # Make the dataset
    dataset = Dataset(args.dataset_name)

    create_index(inputs=dataset.get_val_inputs(), path='../data/{}/val.ann'.format(args.dataset_name))
    create_index(inputs=dataset.get_test_inputs(), path='../data/{}/test.ann'.format(args.dataset_name))
