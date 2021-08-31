import tensorflow as tf2
import numpy as np
import h5py

from dataset.dataset import Dataset
from neural_network.pretrained.vggcifar10 import VggCifar10


if __name__ == '__main__':
    model = VggCifar10()

    dataset = Dataset(dataset_name='cifar_10')

    X = dataset._val_inputs
    y = dataset._val_labels

    probs = model.predict(X, batch_size=32)
    preds = np.argmax(probs, axis=-1)

    accuracy = np.isclose(preds, y)
    print('Accuracy: {}'.format(np.average(accuracy)))

    with h5py.File('neural_network/pretrained/vggcifar10_val.h5', 'w') as fout:
        probs_dataset = fout.create_dataset('probs', dtype='f', shape=probs.shape)
        probs_dataset.write_direct(probs)

        preds_dataset = fout.create_dataset('preds', dtype='i', shape=preds.shape)
        preds_dataset.write_direct(preds)
