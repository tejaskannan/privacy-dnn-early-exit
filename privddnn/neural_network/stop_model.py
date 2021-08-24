import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np
import os.path
import math
from datetime import datetime
from typing import Dict, Tuple, List

from .base import NeuralNetwork
from .constants import PhName, OpName, TRAIN_FRAC, BATCH_SIZE, NUM_EPOCHS, DROPOUT_KEEP_RATE, EARLY_STOP_PATIENCE, STOP_RATES
from privddnn.utils.file_utils import save_json_gz, make_dir


class StopNeuralNetwork(NeuralNetwork):

    def make_placeholders(self, input_shape: Tuple[int, ...]) -> Dict[PhName, tf1.placeholder]:
        """
        Creates the placeholders for this model.

        Returns:
            A dictionary of placeholder name -> placeholder 'tensor'
        """
        placeholders = super().make_placeholders(input_shape=input_shape)

        loss_weight_ph = tf1.placeholder(shape=(),
                                         dtype=tf1.float32,
                                         name=PhName.LOSS_WEIGHT.name.lower())
        placeholders[PhName.LOSS_WEIGHT] = loss_weight_ph

        correct_ph = tf1.placeholder(shape=(None, 2),
                                     dtype=tf1.float32,
                                     name=PhName.MODEL_CORRECT.name.lower())
        placeholders[PhName.MODEL_CORRECT] = correct_ph

        target_rates_ph = tf1.placeholder(shape=(2, ),
                                          dtype=tf1.float32,
                                          name=PhName.STOP_RATES.name.lower())
        placeholders[PhName.STOP_RATES] = target_rates_ph

        return placeholders

    def batch_to_feed_dict(self, inputs: np.ndarray, labels: np.ndarray, model_correct: np.ndarray, dropout_keep_rate: float) -> Dict[tf1.placeholder, np.ndarray]:
        """
        Converts the batch inputs into a feed dict for tensorflow.
        """
        feed_dict = super().batch_to_feed_dict(inputs=inputs,
                                               labels=labels,
                                               dropout_keep_rate=dropout_keep_rate)

        feed_dict[self.placeholders[PhName.LOSS_WEIGHT]] = 10.0
        feed_dict[self.placeholders[PhName.MODEL_CORRECT]] = model_correct
        feed_dict[self.placeholders[PhName.STOP_RATES]] = self.hypers[STOP_RATES]

        return feed_dict

    def train(self, inputs: np.ndarray, labels: np.ndarray, model_correct: np.ndarray, save_folder: str):
        """
        Trains the neural network on the given data.
        """
        assert inputs.shape[0] == labels.shape[0], 'Must provide same number of inputs ({}) as labels ({})'.format(inputs.shape[0], labels.shape[0])

        # Split the training inputs into train / validation folds
        num_samples = inputs.shape[0]
        sample_idx = np.arange(num_samples)
        self._rand.shuffle(sample_idx)

        split_idx = int(num_samples * self.hypers[TRAIN_FRAC])
        train_idx = sample_idx[:split_idx]
        val_idx = sample_idx[split_idx:]

        train_inputs, train_labels, train_correct = inputs[train_idx], labels[train_idx], model_correct[train_idx]
        val_inputs, val_labels, val_correct = inputs[val_idx], labels[val_idx], model_correct[val_idx]

        # Load the metadata for these inputs
        self.load_metadata(train_inputs=train_inputs, train_labels=train_labels)

        # Make the model
        self.make()
        self.init()

        # Make the save folder and get the file name based on the current time
        current_time = datetime.now()
        model_name = '{}_{}'.format(self.name, current_time.strftime('%d-%m-%Y-%H-%M-%S'))

        make_dir(save_folder)

        save_folder = os.path.join(save_folder, current_time.strftime('%d-%m-%Y'))
        make_dir(save_folder)

        save_path = os.path.join(save_folder, '{}.pkl.gz'.format(model_name))

        # Make indices for batch creation
        batch_size = self.hypers[BATCH_SIZE]
        num_epochs = self.hypers[NUM_EPOCHS]

        num_train = train_inputs.shape[0]
        num_val = val_inputs.shape[0]
        train_idx = np.arange(num_train)

        train_ops = [OpName.OPTIMIZE, OpName.LOSS]
        val_ops = [OpName.LOSS]

        num_train_batches = int(math.ceil(num_train / batch_size))
        num_val_batches = int(math.ceil(num_val / batch_size))

        best_val_loss = 1e7
        num_not_improved = 0

        train_log: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(num_epochs):
            print('===== Epoch {} ====='.format(epoch))

            # Execute Model Training
            self._rand.shuffle(train_idx)

            train_loss = 0.0
            num_train_samples = 0.0

            for batch_num, start in enumerate(range(0, num_train, batch_size)):
                # Make the batch
                batch_idx = train_idx[start:start+batch_size]
                batch_inputs = train_inputs[batch_idx]
                batch_labels = train_labels[batch_idx]
                batch_correct = train_correct[batch_idx]

                feed_dict = self.batch_to_feed_dict(inputs=batch_inputs,
                                                    labels=batch_labels,
                                                    model_correct=batch_correct,
                                                    dropout_keep_rate=self.hypers[DROPOUT_KEEP_RATE])

                # Execute the neural network
                train_results = self.execute(feed_dict=feed_dict, ops=train_ops)

                # Compute the statistics
                train_loss += train_results[OpName.LOSS]
                num_train_samples += len(batch_inputs)

                if ((batch_num + 1) % 20 == 0) or (batch_num == (num_train_batches - 1)):
                    avg_train_loss = train_loss / num_train_samples
                    print('Train Batch {}/{}. Loss: {:.4f}'.format(batch_num + 1, num_train_batches, avg_train_loss), end='\r')

            print()

            epoch_train_loss = train_loss / num_train_samples

            # Execute Model Validation
            val_loss = 0.0
            num_val_samples = 0.0

            for batch_num, start in enumerate(range(0, num_val, batch_size)):
                # Make the batch
                batch_inputs = val_inputs[start:start+batch_size]
                batch_labels = val_labels[start:start+batch_size]
                batch_correct = val_correct[start:start+batch_size]

                feed_dict = self.batch_to_feed_dict(inputs=batch_inputs,
                                                    labels=batch_labels,
                                                    model_correct=batch_correct,
                                                    dropout_keep_rate=1.0)

                # Execute the neural network
                val_results = self.execute(feed_dict=feed_dict, ops=val_ops)

                # Compute the statistics
                val_loss += val_results[OpName.LOSS]
                num_val_samples += len(batch_inputs)

                if ((batch_num + 1) % 20 == 0) or (batch_num == (num_val_batches - 1)):
                    avg_val_loss = val_loss / num_val_samples
                    print('Val Batch {}/{}, Loss: {:.4f}'.format(batch_num + 1, num_val_batches, avg_val_loss), end='\r')

            print()

            epoch_val_loss = val_loss / num_val_samples

            train_log['train_loss'].append(epoch_train_loss)
            train_log['val_loss'].append(epoch_val_loss)

            if (epoch_val_loss < best_val_loss):
                best_val_loss = epoch_val_loss
                num_not_improved = 0
                self.save(path=save_path)
                print('Saving...')
            else:
                num_not_improved += 1

            if num_not_improved >= self.hypers[EARLY_STOP_PATIENCE]:
                print('Quitting due to early stoping')
                break

        # Save the training log
        train_log_path = os.path.join(save_folder, '{}_train-log.json.gz'.format(model_name))
        save_json_gz(train_log, train_log_path)

        return save_path
