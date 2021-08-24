import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np
import math
import os.path
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Union, Any, Tuple, List

from privddnn.utils.file_utils import make_dir, save_json_gz, save_pickle_gz, read_pickle_gz
from .constants import OpName, PhName, MetaName, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, DECAY_PATIENCE
from .constants import LEARNING_RATE_DECAY, GRADIENT_CLIP, EARLY_STOP_PATIENCE, TRAIN_FRAC, DROPOUT_KEEP_RATE


class NeuralNetwork:

    def __init__(self, hypers: OrderedDict):
        # Load the default hyper-parameters and override values when needed
        self._hypers = self.default_hypers
        self._hypers.update(**hypers)

        # Initialize operations and placeholders
        self._ops: Dict[OpName, tf2.Tensor] = dict()
        self._placeholders: Dict[PhName, tf1.placeholder] = dict()
        self._metadata: Dict[MetaName, Any] = dict()

        # Create the tensorflow session
        self._sess = tf1.Session(graph=tf2.Graph())

        # Variables to track state of model creation
        self._is_metadata_loaded = False
        self._is_model_made = False

        # Set random seeds to ensure reproducible results
        self._rand = np.random.RandomState(seed=23789)

        with self._sess.graph.as_default():
            tf2.random.set_seed(389313)

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def default_hypers(self) -> Dict[str, Union[float, int]]:
        return {
            NUM_EPOCHS: 2,
            BATCH_SIZE: 32,
            TRAIN_FRAC: 0.8,
            LEARNING_RATE: 0.001,
            LEARNING_RATE_DECAY: 0.9,
            GRADIENT_CLIP: 1.0,
            EARLY_STOP_PATIENCE: 2,
            DECAY_PATIENCE: 2,
            DROPOUT_KEEP_RATE: 0.8
        }

    @property
    def hypers(self) -> Dict[str, Any]:
        return self._hypers

    @property
    def ops(self) -> Dict[OpName, tf2.Tensor]:
        return self._ops

    @property
    def placeholders(self) -> Dict[PhName, tf1.placeholder]:
        return self._placeholders

    @property
    def metadata(self) -> Dict[MetaName, Any]:
        return self._metadata

    @property
    def sess(self) -> tf1.Session:
        return self._sess

    @property
    def is_metadata_loaded(self) -> bool:
        return self._is_metadata_loaded

    @property
    def is_model_made(self) -> bool:
        return self._is_model_made

    def make_placeholders(self, input_shape: Tuple[int, ...]) -> Dict[PhName, tf1.placeholder]:
        """
        Creates the placeholders for this model.

        Returns:
            A dictionary of placeholder name -> placeholder 'tensor'
        """
        inputs_ph = tf1.placeholder(shape=(None,) + input_shape,
                                    dtype=tf2.float32,
                                    name=PhName.INPUTS.name.lower())

        labels_ph = tf1.placeholder(shape=(None,),
                                    dtype=tf2.int32,
                                    name=PhName.LABELS.name.lower())

        dropout_keep_rate_ph = tf1.placeholder(shape=(),
                                               dtype=tf2.float32,
                                               name=PhName.DROPOUT_KEEP_RATE.name.lower())

        return {
            PhName.INPUTS: inputs_ph,
            PhName.LABELS: labels_ph,
            PhName.DROPOUT_KEEP_RATE: dropout_keep_rate_ph
        }

    def make_model(self, inputs: tf1.placeholder, dropout_keep_rate: tf1.placeholder, num_labels: int) -> tf2.Tensor:
        """
        Creates the computational graph.

        Args:
            inputs: A placeholder ([B, ...]) representing the input values
            dropout_keep_rate: A placeholder holding the dropout keep rate
            num_labels: The number of labels (K)
        Returns:
            The raw logits for this model, generally of size [B, K]
        """
        raise NotImplementedError()

    def make_loss(self, logits: tf2.Tensor, labels: tf1.placeholder) -> tf2.Tensor:
        """
        Creates the loss function for this model.

        Args:
            logits: The log probabilities, generally [B, K]
            labels: The labels, [B]
        """
        raise NotImplementedError()

    def make_optimizer_op(self, loss: tf2.Tensor) -> tf2.Tensor:
        """
        Makes the optimizer operation for this model.

        Args:
            loss: A scaler tensor representing the loss
        Returns:
            An operation that performs gradient updates
        """
        # Get the collection of trainable variables
        trainable_vars = self.sess.graph.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES)

        # Compute the gradients
        gradients = tf2.gradients(loss, trainable_vars)

        # Clip Gradients
        clipped_gradients, _ = tf2.clip_by_global_norm(gradients, self.hypers[GRADIENT_CLIP])

        # Prune None values from the set of gradients and apply gradient weights
        pruned_gradients = [(grad, var) for grad, var in zip(clipped_gradients, trainable_vars) if grad is not None]

        # Apply clipped gradients
        self._optimizer = tf1.train.AdamOptimizer(learning_rate=self.hypers[LEARNING_RATE])
        optimizer_op = self._optimizer.apply_gradients(pruned_gradients)

        return optimizer_op

    def make(self):
        """
        Builds this model.
        """
        assert self.is_metadata_loaded, 'Must load metadata before making the model.'

        if self.is_model_made:
            return

        with self.sess.graph.as_default():
            # Make the placeholders
            self._placeholders = self.make_placeholders(input_shape=self._metadata[MetaName.INPUT_SHAPE])

            # Make the computational graph
            logits = self.make_model(inputs=self._placeholders[PhName.INPUTS],
                                     dropout_keep_rate=self._placeholders[PhName.DROPOUT_KEEP_RATE],
                                     num_labels=self.metadata[MetaName.NUM_LABELS])

            # Make the loss function
            loss = self.make_loss(logits=logits,
                                  labels=self._placeholders[PhName.LABELS])

            # Make the optimization step
            optimizer_op = self.make_optimizer_op(loss=loss)

            # Get the predictions
            predictions = tf2.argmax(logits, axis=-1)
            probs = tf2.nn.softmax(logits, axis=-1)

            # Set the operations
            ops = {
                OpName.LOGITS: logits,
                OpName.LOSS: loss,
                OpName.OPTIMIZE: optimizer_op,
                OpName.PREDICTIONS: predictions,
                OpName.PROBS: probs
            }

            for op_name, op_value in ops.items():
                self._ops[op_name] = op_value

        self._is_model_made = True

    def batch_to_feed_dict(self, inputs: np.ndarray, labels: np.ndarray, dropout_keep_rate: float) -> Dict[tf1.placeholder, np.ndarray]:
        """
        Converts the batch inputs into a feed dict for tensorflow.
        """
        return {
            self.placeholders[PhName.INPUTS]: inputs,
            self.placeholders[PhName.LABELS]: labels,
            self.placeholders[PhName.DROPOUT_KEEP_RATE]: dropout_keep_rate
        }

    def init(self):
        """
        Initializes the variables in the given computational graph.
        """
        with self.sess.graph.as_default():
            self.sess.run(tf1.global_variables_initializer())

    def execute(self, feed_dict: Dict[tf1.placeholder, np.ndarray], ops: List[OpName]) -> Dict[OpName, np.ndarray]:
        """
        Executes the given operations for this neural network.
        """
        with self.sess.graph.as_default():
            ops_to_run = { name.name.upper(): self.ops[name] for name in ops }
            results = self.sess.run(ops_to_run, feed_dict=feed_dict)

        return { OpName[name]: value for name, value in results.items() }


    def load_metadata(self, train_inputs: np.ndarray, train_labels: np.ndarray):
        """
        Extracts the input shape and fits the data scaler
        """
        # Prevent loading metadata twice
        if self.is_metadata_loaded:
            return

        # Fit the data scaler
        #reshaped_inputs = train_inputs.reshape(train_inputs.shape[0], -1)
        #mean = np.average(reshaped_inputs, axis=0)  # [D]
        #std = np.std(reshaped_inputs, axis=0)  # [D]
        input_max = np.max(train_inputs)
        input_min = np.min(train_inputs)

        # Get the number of labels
        num_labels = np.amax(train_labels) + 1

        # Save the metadata
        self._metadata[MetaName.INPUT_SHAPE] = train_inputs.shape[1:]
        self._metadata[MetaName.INPUT_MEAN] = input_max
        self._metadata[MetaName.INPUT_STD] = input_min
        self._metadata[MetaName.NUM_LABELS] = num_labels

        self._is_metadata_loaded = True

    def normalize_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """
        Normalizes the given inputs
        """
        assert self.is_metadata_loaded, 'Must first load the metadata'

        input_max = self._metadata[MetaName.INPUT_MEAN]
        input_min = self._metadata[MetaName.INPUT_STD]

        return (inputs - input_min) / (input_max - input_min)

        #input_shape = inputs.shape
        #reshaped_inputs = inputs.reshape(input_shape[0], -1)
        #scaled_inputs = (reshaped_inputs - self.metadata[MetaName.INPUT_MEAN]) / (self.metadata[MetaName.INPUT_STD])

        #return scaled_inputs.reshape(input_shape)

    def train(self, inputs: np.ndarray, labels: np.ndarray, save_folder: str):
        """
        Trains the neural network on the given data.
        """
        assert inputs.shape[0] == labels.shape[0], 'Must provide same number of inputs ({}) as labels ({})'.format(inputs.shape[0], labels.shape[0])

        # Split the training inputs into train / validation folds
        num_samples = inputs.shape[0]
        sample_idx = np.arange(num_samples)
        self._rand.shuffle(sample_idx)

        split_idx = int(num_samples * self.hypers[TRAIN_FRAC])
        train_idx, val_idx = sample_idx[:split_idx], sample_idx[split_idx:]

        train_inputs, train_labels = inputs[train_idx], labels[train_idx]
        val_inputs, val_labels = inputs[val_idx], labels[val_idx]

        # Load the metadata for these inputs
        self.load_metadata(train_inputs=train_inputs, train_labels=train_labels)

        # Scale the data
        train_inputs = self.normalize_inputs(train_inputs)
        val_inputs = self.normalize_inputs(val_inputs)

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

        train_ops = [OpName.OPTIMIZE, OpName.LOSS, OpName.PREDICTIONS]
        val_ops = [OpName.LOSS, OpName.PREDICTIONS]

        num_train_batches = int(math.ceil(num_train / batch_size))
        num_val_batches = int(math.ceil(num_val / batch_size))

        best_val_accuracy = 0.0
        num_not_improved = 0

        train_log: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

        for epoch in range(num_epochs):
            print('===== Epoch {} ====='.format(epoch))

            # Execute Model Training
            self._rand.shuffle(train_idx)

            train_correct = 0.0
            train_loss = 0.0
            num_train_samples = 0.0

            for batch_num, start in enumerate(range(0, num_train, batch_size)):
                # Make the batch
                batch_idx = train_idx[start:start+batch_size]
                batch_inputs = train_inputs[batch_idx]
                batch_labels = train_labels[batch_idx]

                feed_dict = self.batch_to_feed_dict(inputs=batch_inputs,
                                                    labels=batch_labels,
                                                    dropout_keep_rate=self.hypers[DROPOUT_KEEP_RATE])

                # Execute the neural network
                train_results = self.execute(feed_dict=feed_dict, ops=train_ops)

                # Expand labels if needed
                train_pred = train_results[OpName.PREDICTIONS]
                num_outputs = 1

                if len(train_pred.shape) == 2:
                    batch_labels = np.expand_dims(batch_labels, axis=-1)
                    num_outputs = train_pred.shape[-1]

                # Compute the statistics
                train_loss += train_results[OpName.LOSS]
                train_correct += np.sum(np.isclose(train_pred, batch_labels))
                num_train_samples += len(batch_inputs)

                avg_train_loss = train_loss / num_train_samples
                train_accuracy = train_correct / (num_train_samples * num_outputs)

                if ((batch_num + 1) % 20 == 0) or (batch_num == (num_train_batches - 1)):
                    print('Train Batch {}/{}. Loss: {:.4f}, Accuracy: {:.4f}'.format(batch_num + 1, num_train_batches, avg_train_loss, train_accuracy), end='\r')

            print()

            epoch_train_loss = train_loss / num_train_samples
            epoch_train_acc = train_correct / (num_train_samples * num_outputs)

            # Execute Model Validation
            val_correct = 0.0
            val_loss = 0.0
            num_val_samples = 0.0

            for batch_num, start in enumerate(range(0, num_val, batch_size)):
                # Make the batch
                batch_inputs = val_inputs[start:start+batch_size]
                batch_labels = val_labels[start:start+batch_size]

                feed_dict = self.batch_to_feed_dict(inputs=batch_inputs,
                                                    labels=batch_labels,
                                                    dropout_keep_rate=1.0)

                # Execute the neural network
                val_results = self.execute(feed_dict=feed_dict, ops=val_ops)

                # Expand labels if needed
                val_pred = val_results[OpName.PREDICTIONS]
                num_outputs = 1

                if len(val_pred.shape) == 2:
                    batch_labels = np.expand_dims(batch_labels, axis=-1)
                    num_outputs = val_pred.shape[-1]

                # Compute the statistics
                val_loss += val_results[OpName.LOSS]
                val_correct += np.sum(np.isclose(val_pred, batch_labels))
                num_val_samples += len(batch_inputs)

                avg_val_loss = val_loss / num_val_samples
                val_accuracy = val_correct / (num_val_samples * num_outputs)

                if ((batch_num + 1) % 20 == 0) or (batch_num == (num_val_batches - 1)):
                    print('Val Batch {}/{}, Loss: {:.4f}, Accuracy: {:.4f}'.format(batch_num + 1, num_val_batches, avg_val_loss, val_accuracy), end='\r')

            print()

            epoch_val_loss = val_loss / num_val_samples
            epoch_val_acc = val_correct / (num_val_samples * num_outputs)

            train_log['train_loss'].append(epoch_train_loss)
            train_log['train_accuracy'].append(epoch_train_acc)
            train_log['val_loss'].append(epoch_val_loss)
            train_log['val_accuracy'].append(epoch_val_acc)

            if (epoch_val_acc > best_val_accuracy):
                best_val_accuracy = epoch_val_acc
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

    def predict(self, inputs: np.ndarray, pred_op: OpName) -> np.ndarray:
        """
        Computes the predictions on the given (unscaled) inputs.
        """
        pred_list: List[np.ndarray] = []
        batch_size = self.hypers[BATCH_SIZE]
        num_samples = inputs.shape[0]

        # Scale the inputs
        inputs = self.normalize_inputs(inputs=inputs)

        for start in range(0, num_samples, batch_size):
            batch_inputs = inputs[start:start+batch_size]

            feed_dict = {
                self._placeholders[PhName.INPUTS]: batch_inputs,
                self._placeholders[PhName.DROPOUT_KEEP_RATE]: 1.0
            }

            batch_result = self.execute(feed_dict=feed_dict, ops=[pred_op])
            pred_list.append(batch_result[pred_op])

        return np.concatenate(pred_list, axis=0)

    def save(self, path: str):
        """
        Saves the current model paramters in the given (pickle) file.
        """
        output_data: Dict[str, Any] = dict()

        # Save the metadata and hyper-parameters
        output_data['metadata'] = self.metadata
        output_data['hypers'] = self.hypers

        # Save the trainable variables
        with self.sess.graph.as_default():
            trainable_vars = self.sess.graph.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES)
            var_dict = { var.name: var for var in trainable_vars }
            var_values = self.sess.run(var_dict)

            output_data['weights'] = var_values

        save_pickle_gz(output_data, path)

    @classmethod
    def restore(cls, path: str):
        """
        Restores the model saved at the given path
        """
        # Load the saved data
        serialized_data = read_pickle_gz(path)

        hypers = serialized_data['hypers']
        metadata = serialized_data['metadata']
        model_weights = serialized_data['weights']

        # Build the model
        model = cls(hypers=hypers)

        model._metadata = metadata
        model._is_metadata_loaded = True

        model.make()
        model.init()

        # Set the model weights
        with model.sess.graph.as_default():
            trainable_vars = model.sess.graph.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES)
            var_dict = { var.name: var for var in trainable_vars }

            assign_ops = [tf1.assign(var_dict[name], model_weights[name]) for name in model_weights.keys()]
            model.sess.run(assign_ops)

        return model
