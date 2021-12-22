import numpy as np
from annoy import AnnoyIndex
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from typing import Optional

from privddnn.dataset.dataset import Dataset
from privddnn.utils.file_utils import save_pickle_gz


NUM_COMPONENTS = 32
NUM_TREES = 16


def fit_pca(inputs: np.ndarray, n_components: int) -> Optional[PCA]:
    flattened = inputs.reshape(inputs.shape[0], -1)
    num_features = flattened.shape[-1]

    if num_features < n_components:
        return None

    pca = PCA(n_components=n_components)
    pca.fit(flattened)

    return pca


def create_index(inputs: np.ndarray, pca: Optional[PCA], path: str):
    num_samples = inputs.shape[0]
    flattened = inputs.reshape(num_samples, -1)

    if pca is not None:
        flattened = pca.transform(flattened)

    num_features = flattened.shape[-1]
    index = AnnoyIndex(num_features, 'euclidean')

    for idx, features in enumerate(flattened):
        index.add_item(idx, features)

    index.build(NUM_TREES)
    index.save(path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    args = parser.parse_args()

    # Make the dataset
    dataset = Dataset(args.dataset_name)

    pca = fit_pca(inputs=dataset.get_val_inputs(), n_components=NUM_COMPONENTS)

    if pca is not None:
        save_pickle_gz(pca, '../data/{}/pca.pkl.gz'.format(args.dataset_name))

    create_index(inputs=dataset.get_val_inputs(), pca=pca, path='../data/{}/val.ann'.format(args.dataset_name))
    create_index(inputs=dataset.get_test_inputs(), pca=pca, path='../data/{}/test.ann'.format(args.dataset_name))
