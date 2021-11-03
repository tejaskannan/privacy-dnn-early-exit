import os.path
from argparse import ArgumentParser
from privddnn.dataset.dataset import Dataset, DataFold
from privddnn.serialize.utils import array_to_fixed_point
from privddnn.utils.file_utils import make_dir


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True, help='The name of the dataset to serialize.')
    parser.add_argument('--precision', type=int, required=True, help='The fixed point precision.')
    parser.add_argument('--fold', type=str, required=True, choices=['train', 'val', 'test'], help='The dataset fold to serialize.')
    args = parser.parse_args()

    assert (args.precision >= 0) and (args.precision <= 15), 'The precision must be in [0, 16).'

    dataset = Dataset(dataset_name=args.dataset_name)

    # Normalize the values
    dataset.fit_normalizer(is_global=False)
    dataset.normalize_data()

    fold = DataFold[args.fold.upper()]
    inputs = dataset.get_inputs(fold=fold)
    labels = dataset.get_labels(fold=fold)

    dataset_folder = '../data/{}'.format(args.dataset_name)
    make_dir(dataset_folder)

    inputs_path = os.path.join(dataset_folder, '{}_{}_inputs.txt'.format(args.dataset_name, args.precision))
    labels_path = os.path.join(dataset_folder, '{}_{}_labels.txt'.format(args.dataset_name, args.precision))

    # Quantize the input features
    quantized_inputs = array_to_fixed_point(inputs, precision=args.precision, width=16)

    with open(inputs_path, 'w') as fout:
        for feature_vector in quantized_inputs:
            feature_str = list(map(str, feature_vector))
            fout.write(' '.join(feature_str) + '\n')

    with open(labels_path, 'w') as fout:
        for label in labels:
            fout.write('{}\n'.format(label))
