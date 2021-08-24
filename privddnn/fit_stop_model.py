import tensorflow as tf2
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from enum import Enum, auto
from typing import List, Tuple, DefaultDict

from neural_network import restore_model, NeuralNetwork, OpName
from neural_network.stop_dnn import StopDNN
from exiting.early_exit import entropy_exit
from privddnn.utils.metrics import compute_accuracy, compute_mutual_info



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--output-path', type=str, help='Path to save the final plot')
    args = parser.parse_args()

    # Restore the model
    model: NeuralNetwork = restore_model(path=args.model_path)

    # Get the dataset
    (train_inputs, train_labels), _ = tf2.keras.datasets.cifar10.load_data()
    train_labels = train_labels.reshape(-1)  # [B]

    # Get the predictions from the models
    model_preds = model.predict(inputs=train_inputs, pred_op=OpName.PREDICTIONS)  # [B, K]
    model_states = model.predict(inputs=train_inputs, pred_op=OpName.STATE)  # [B, K, D]

    model_correct = np.isclose(np.expand_dims(train_labels, axis=-1), model_preds).astype(int)  # [B, K]

    stop_model = StopDNN(hypers=dict(num_epochs=50, stop_rates=[0.5, 0.5], early_stop_patience=10, batch_size=256))
    stop_model.train(inputs=model_states, labels=train_labels, model_correct=model_correct, save_folder='saved_models')

    #stop_model = StopDNN.restore('saved_models/24-08-2021/stop-dnn_24-08-2021-13-19-00.pkl.gz')
    #stop_probs = stop_model.predict(inputs=model_states, pred_op=OpName.PROBS)  # [B, L]

    #output_indices = defaultdict(list)
    #num_correct = 0.0
    #num_total = 0.0

    #rand = np.random.RandomState(seed=7894)
    #for idx in range(len(test_labels)):
    #    label = test_labels[idx]
    #    stop_rates = stop_probs[idx]

    #    if idx < 10:
    #        print('Label: {}, Stop Rates: {}'.format(label, stop_rates))

    #    r = rand.uniform()
    #    stop_idx = int(r > stop_rates[0])

    #    num_correct += model_correct[idx, stop_idx]
    #    num_total += 1.0

    #    output_indices[label].append(stop_idx)

    #for label, stops in sorted(output_indices.items()):
    #    avg = np.average(stops)
    #    std = np.std(stops)
    #    print('{} -> {}, {}'.format(label, avg, std))

    #print('Accuracy: {}'.format(num_correct / num_total))
