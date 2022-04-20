import os.path
import h5py
from argparse import ArgumentParser
from sklearn.decomposition import PCA

from privddnn.utils.file_utils import read_json_gz, save_json_gz, make_dir
from privddnn.utils.loading import load_h5_dataset



def fit_pca(input_file: str, num_dims: int) -> PCA:
    inputs, _ = load_h5_dataset(input_file)

    pca = PCA(n_components=num_dims)
    pca.fit(inputs)

    return pca


def convert_fold(input_file: str, output_file: str, pca: PCA):
    inputs, labels = load_h5_dataset(input_file)

    assert len(inputs.shape) == 2, 'Must provide a 2d matrix of inputs. Found shape: {}'.format(inputs.shape)

    reduced_inputs = pca.transform(inputs)

    with h5py.File(output_file, 'w') as fout:
        input_ds = fout.create_dataset('inputs', shape=reduced_inputs.shape, dtype='f')
        input_ds.write_direct(reduced_inputs)

        label_ds = fout.create_dataset('labels', shape=labels.shape, dtype='i')
        label_ds.write_direct(labels)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--old-dataset', type=str, required=True, help='Name of the existing (sequential) dataset')
    parser.add_argument('--new-dataset', type=str, required=True, help='Name of the new dataset')
    parser.add_argument('--num-dims', type=int, required=True, help='Number of dimensions in the new dataset.')
    args = parser.parse_args()

    input_dir = os.path.join('..', 'data', args.old_dataset)
    output_dir = os.path.join('..', 'data', args.new_dataset)

    # Make the output dataset folder (if needed)
    make_dir(output_dir)

    pca = fit_pca(input_file=os.path.join(input_dir, 'train.h5'),
                  num_dims=args.num_dims)

    for fold in ['train', 'val', 'test']:
        convert_fold(input_file=os.path.join(input_dir, '{}.h5').format(fold),
                     output_file=os.path.join(output_dir, '{}.h5').format(fold),
                     pca=pca)


