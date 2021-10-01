import tensorflow as tf2
import tensorflow.compat.v1 as tf1
import numpy as np
import math
import os.path
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Union, Any, Tuple, List, Iterable

from privddnn.utils.file_utils import make_dir, save_json_gz, save_pickle_gz, read_pickle_gz
from privddnn.classifier import OpName, ModelMode, BaseClassifier
from .constants import PhName, MetaName, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, DECAY_PATIENCE
from .constants import LEARNING_RATE_DECAY, GRADIENT_CLIP, EARLY_STOP_PATIENCE, TRAIN_FRAC, DROPOUT_KEEP_RATE, STOP_RATES


class NeuralNetwork(BaseClassifier):

    def __init__(self, dataset_name: str, hypers: OrderedDict):
        super().__init__(dataset_name=dataset_name)

        # Load the default hyper-parameters and override values when needed
        self._hypers = self.default_hypers
        self._hypers.update(**hypers)

        self._learning_rate = self._hypers[LEARNING_RATE]

        # Initialize operations and placeholders
        self._ops: Dict[OpName, tf2.Tensor] = dict()
        self._placeholders: Dict[PhName, tf1.placeholder] = dict()
        self._metadata: Dict[MetaName, Any] = dict()

        # Create the tensorflow session
        self._sess = tf1.Session(graph=tf2.Graph())

        # Variables to track state of model creation
        self._is_metadata_loaded = False
        self._is_model_made = False
        self._is_init = False

        # Set random seeds to ensure reproducible results
        self._rand = np.random.RandomState(seed=23789)

        with self._sess.graph.as_default():
            tf2.random.set_seed(389313)

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def default_hypers(self) -> Dict[str, Any]:
        return {
            NUM_EPOCHS: 2,
            BATCH_SIZE: 32,
            LEARNING_RATE: 0.001,
            LEARNING_RATE_DECAY: 0.9,
            GRADIENT_CLIP: 1.0,
            EARLY_STOP_PATIENCE: 2,
            DECAY_PATIENCE: 2,
            DROPOUT_KEEP_RATE: 0.8,
            STOP_RATES: [0.5, 0.5]
        }

    @property
    def hypers(self) -> Dict[str, Any]:
        return self._hypers

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

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

    @property
    def is_init(self) -> bool:
        return self._is_init

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

        learning_rate_ph = tf1.placeholder(shape=(),
                                           dtype=tf2.float32,
                                           name=PhName.LEARNING_RATE.name.lower())
        return {
            PhName.INPUTS: inputs_ph,
            PhName.LABELS: labels_ph,
            PhName.DROPOUT_KEEP_RATE: dropout_keep_rate_ph,
            PhName.LEARNING_RATE: learning_rate_ph
        }

    def make_model(self, inputs: tf1.placeholder, dropout_keep_rate: tf1.placeholder, num_labels: int, model_mode: ModelMode) -> tf2.Tensor:
        """
        Creates the computational graph.

        Args:
            inputs: A placeholder ([B, ...]) representing the input values
            dropout_keep_rate: A placeholder holding the dropout keep rate
            num_labels: The number of labels (K)
            model_mode: The mode (train or fine tune)
        Returns:
            The raw logits for this model, generally of size [B, K]
        """
        raise NotImplementedError()

    def make_loss(self, logits: tf2.Tensor, labels: tf1.placeholder, model_mode: ModelMode) -> tf2.Tensor:
        """
        Creates the loss function for this model.

        Args:
            logits: The log probabilities, generally [B, K]
            labels: The labels, [B]
            model_mode: The mode (train or fine tune)
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
        pruned_gradients = []
        for grad, var in zip(clipped_gradients, trainable_vars):
            if grad is not None:
                pruned_gradients.append((grad, var))
            else:
                print('WARNING: Gradient is None for {}'.format(var.name))

        # Apply clipped gradients
        self._optimizer = tf1.train.AdamOptimizer(learning_rate=self._placeholders[PhName.LEARNING_RATE])
        optimizer_op = self._optimizer.apply_gradients(pruned_gradients)

        return optimizer_op

    def make(self, model_mode: ModelMode):
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
                                     num_labels=self.metadata[MetaName.NUM_LABELS],
                                     model_mode=model_mode)

            # Make the loss function
            loss = self.make_loss(logits=logits,
                                  labels=self._placeholders[PhName.LABELS],
                                  model_mode=model_mode)

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
            self.placeholders[PhName.DROPOUT_KEEP_RATE]: dropout_keep_rate,
            self.placeholders[PhName.LEARNING_RATE]: self.learning_rate
        }

    def init(self):
        """
        Initializes the variables in the given computational graph.
        """
        if self.is_init:
            return

        with self.sess.graph.as_default():
            self.sess.run(tf1.global_variables_initializer())

        self._is_init = True

    def execute(self, feed_dict: Dict[tf1.placeholder, np.ndarray], ops: List[OpName]) -> Dict[OpName, np.ndarray]:
        """
        Executes the given operations for this neural network.
        """
        with self.sess.graph.as_default():
            ops_to_run = { name.name.upper(): self.ops[name] for name in ops }
            results = self.sess.run(ops_to_run, feed_dict=feed_dict)

        return { OpName[name]: value for name, value in results.items() }


    def load_metadata(self):
        """
        Extracts the input shape and fits the data scaler
        """
        # Prevent loading metadata twice
        if self.is_metadata_loaded:
            return

        # Normalize the data
        self.dataset.fit_normalizer(is_global=True)
        self.dataset.normalize_data()

        # Save the metadata
        self._metadata[MetaName.INPUT_SHAPE] = self.dataset.input_shape
        self._metadata[MetaName.NUM_LABELS] = self.dataset.num_labels
        self._metadata[MetaName.DATASET_NAME] = self.dataset.dataset_name

        self._is_metadata_loaded = True

    def train(self, save_folder: str, model_mode: ModelMode):
        """
        Trains the neural network on the given data.
        """
        # Load the metadata for these inputs
        self.load_metadata()

        # Make the model
        self.make(model_mode=model_mode)
        self.init()

        # Make the save folder and get the file name based on the current time
        current_time = datetime.now()
        model_name = '{}_{}'.format(self.name, current_time.strftime('%d-%m-%Y-%H-%M-%S'))

        make_dir(save_folder)
        save_folder = os.path.join(save_folder, self.dataset.dataset_name)
        make_dir(save_folder)

        save_folder = os.path.join(save_folder, current_time.strftime('%d-%m-%Y'))
        make_dir(save_folder)

        save_path = os.path.join(save_folder, '{}.pkl.gz'.format(model_name))

        # Make indices for batch creation
        batch_size = self.hypers[BATCH_SIZE]
        num_epochs = self.hypers[NUM_EPOCHS]

        num_train = self.dataset.num_train
        num_val = self.dataset.num_val

        train_ops = [OpName.OPTIMIZE, OpName.LOSS, OpName.PREDICTIONS]
        val_ops = [OpName.LOSS, OpName.PREDICTIONS]

        num_train_batches = int(math.ceil(num_train / batch_size))
        num_val_batches = int(math.ceil(num_val / batch_size))

        best_val_accuracy = 0.0
        best_val_loss = 1e7
        num_not_improved = 0
        num_not_improved_lr = 0

        train_log: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

        for epoch in range(num_epochs):
            print('===== Epoch {}/{} ====='.format(epoch + 1, num_epochs))

            # Execute Model Training
            train_correct = 0.0
            train_loss = 0.0
            num_train_samples = 0.0

            train_batch_generator = self.dataset.generate_train_batches(batch_size=batch_size)
            for batch_num, (batch_inputs, batch_labels) in enumerate(train_batch_generator):
                # Make the feed dict
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

            val_batch_generator = self.dataset.generate_val_batches(batch_size=batch_size)
            for batch_num, (batch_inputs, batch_labels) in enumerate(val_batch_generator):
                # Make the feed dict
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

            # Check for model improvement and save accordingly
            did_improve = False
            if model_mode == ModelMode.TRAIN:
                did_improve = (epoch_val_acc > best_val_accuracy)
            else:
                did_improve = (epoch_val_loss < best_val_loss)

            if did_improve:
                best_val_accuracy = epoch_val_acc
                best_val_loss = epoch_val_loss
                num_not_improved = 0
                num_not_improved_lr = 0
                self.save(path=save_path)
                print('Saving...')
            else:
                num_not_improved += 1
                num_not_improved_lr += 1

            if num_not_improved >= self.hypers[EARLY_STOP_PATIENCE]:
                print('Quitting due to early stoping')
                break

            if num_not_improved_lr >= self.hypers[DECAY_PATIENCE]:
                self._learning_rate *= self.hypers[LEARNING_RATE_DECAY]
                num_not_improved_lr = 0

        # Save the training log
        train_log_path = os.path.join(save_folder, '{}_train-log.json.gz'.format(model_name))
        save_json_gz(train_log, train_log_path)

        return save_path

    def test(self, op: OpName) -> np.ndarray:
        """
        Runs the given operation on the test set.
        """
        test_batch_generator = self.dataset.generate_test_batches(batch_size=self.hypers[BATCH_SIZE])
        return self.execute_op(data_generator=test_batch_generator, op=op)

    def validate(self, op: OpName) -> np.ndarray:
        """
        Runs the given operation on the validation set.
        """
        val_batch_generator = self.dataset.generate_val_batches(batch_size=self.hypers[BATCH_SIZE])
        return self.execute_op(data_generator=val_batch_generator, op=op)

    def execute_op(self, data_generator: Iterable[Tuple[np.ndarray, np.ndarray]], op: OpName) -> np.ndarray:
        """
        Computes the predictions on the given (unscaled) inputs.
        """
        pred_list: List[np.ndarray] = []
        num_samples = self.dataset.num_test

        for (batch_inputs, _) in data_generator:
            feed_dict = {
                self._placeholders[PhName.INPUTS]: batch_inputs,
                self._placeholders[PhName.DROPOUT_KEEP_RATE]: 1.0
            }

            batch_result = self.execute(feed_dict=feed_dict, ops=[op])
            pred_list.append(batch_result[op])

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
    def restore(cls, path: str, model_mode: ModelMode):
        """
        Restores the model saved at the given path
        """
        # Load the saved data
        serialized_data = read_pickle_gz(path)

        hypers = serialized_data['hypers']
        metadata = serialized_data['metadata']
        model_weights = serialized_data['weights']

        # Build the model
        model = cls(hypers=hypers, dataset_name=metadata[MetaName.DATASET_NAME])

        model._metadata = metadata
        model._is_metadata_loaded = True

        model.dataset.fit_normalizer(is_global=True)
        model.dataset.normalize_data()

        model.make(model_mode=model_mode)
        model.init()

        # Set the model weights
        with model.sess.graph.as_default():
            trainable_vars = model.sess.graph.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES)
            var_dict = { var.name: var for var in trainable_vars }

            assign_ops = [tf1.assign(var_dict[name], model_weights[name]) for name in model_weights.keys()]
            model.sess.run(assign_ops)

        return model
