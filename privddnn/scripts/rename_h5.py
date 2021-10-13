import h5py
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    # Load the data
    with h5py.File(args.file, 'r') as fin:
        inputs = fin['inputs'][:]
        labels = fin['labels'][:]

    input_shape = inputs.shape
    inputs = inputs.reshape(input_shape[0], -1)

    # Save the updated file
    with h5py.File(args.file, 'w') as fout:
        input_ds = fout.create_dataset('inputs', shape=inputs.shape, dtype='f')
        input_ds.write_direct(inputs)

        label_ds = fout.create_dataset('labels', shape=labels.shape, dtype='i')
        label_ds.write_direct(labels)
