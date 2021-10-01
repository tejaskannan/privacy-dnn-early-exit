import numpy as np
from annoy import AnnoyIndex
from argparse import ArgumentParser
from sklearn.decomposition import PCA

from privddnn.dataset.dataset import Dataset
from privddnn.utils.file_utils import save_pickle_gz


def create_index(inputs: np.ndarray, n_components: int, path: str):
    num_samples = inputs.shape[0]
    flattened = inputs.reshape(num_samples, -1)

    pca = PCA(n_components=n_components, svd_solver='full')
    projected = pca.fit_transform(flattened)

    index = AnnoyIndex(n_components, 'euclidean')
    for idx, features in enumerate(projected):
        index.add_item(idx, features)

    index.build(50)
    index.save('{}.ann'.format(path))

    metadata = {
        'n_components': n_components,
        'pca': pca
    }
    save_pickle_gz(metadata, '{}.pkl.gz'.format(path))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--num-components', type=int, required=True)
    args = parser.parse_args()

    # Make the dataset
    dataset = Dataset(args.dataset_name)

    create_index(inputs=dataset.get_val_inputs(), n_components=args.num_components, path='../data/{}/val'.format(args.dataset_name))
    create_index(inputs=dataset.get_test_inputs(), n_components=args.num_components, path='../data/{}/test'.format(args.dataset_name))
