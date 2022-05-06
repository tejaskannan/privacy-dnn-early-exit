import numpy as np
import math
import os.path
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import Metric
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Union, Any, Tuple, List, Iterable

from privddnn.utils.file_utils import make_dir, save_json_gz, read_json_gz, save_pickle_gz
from privddnn.utils.np_utils import approx_softmax
from privddnn.utils.constants import SMALL_NUMBER
from privddnn.classifier import OpName, ModelMode, BaseClassifier
from .constants import MetaName, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, DECAY_PATIENCE, ACTIVATION
from .constants import LEARNING_RATE_DECAY, GRADIENT_CLIP, EARLY_STOP_PATIENCE, DROPOUT_KEEP_RATE, STOP_RATES


class NeuralNetwork(BaseClassifier):

    def __init__(self, dataset_name: str, hypers: OrderedDict):
        super().__init__(dataset_name=dataset_name)

        # Load the default hyper-parameters and override values when needed
        self._hypers = self.default_hypers
        self._hypers.update(**hypers)

        # Initialize operations and placeholders
        self._metadata: Dict[MetaName, Any] = dict()

        # Variables to track state of model creation
        self._is_metadata_loaded = False
        self._is_model_made = False
        self._is_init = False

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
            EARLY_STOP_PATIENCE: 2,
            DROPOUT_KEEP_RATE: 0.8
        }

    @property
    def hypers(self) -> Dict[str, Any]:
        return self._hypers

    @property
    def learning_rate(self) -> float:
        return self.hypers[LEARNING_RATE]

    @property
    def metadata(self) -> Dict[MetaName, Any]:
        return self._metadata

    @property
    def is_metadata_loaded(self) -> bool:
        return self._is_metadata_loaded

    @property
    def is_model_made(self) -> bool:
        return self._is_model_made

    def make_model(self, inputs: Input, num_labels: int, model_mode: ModelMode) -> Union[Layer, List[Layer]]:
        """
        Creates the computational graph.

        Args:
            num_labels: The number of labels (K)
            model_mode: The mode (train or test)
        Returns:
            The raw logits for this model, generally of size [B, K]
        """
        raise NotImplementedError()

    def make_loss(self) -> Dict[str, str]:
        """
        Creates the loss function for this model.
        """
        raise NotImplementedError()

    def make_metrics(self) -> Dict[str, Metric]:
        """
        Creates the output metrics for this model.
        """
        raise NotImplementedError()

    def make_loss_weights(self) -> Dict[str, float]:
        """
        Creates the loss weights for this model.
        """
        raise NotImplementedError()

    def make_inputs(self) -> Input:
        """
        Creates the input layer for this model.
        """
        input_shape = self.metadata[MetaName.INPUT_SHAPE]
        return Input(shape=input_shape, name='input')

    def make(self, model_mode: ModelMode):
        """
        Builds this model.
        """
        assert self.is_metadata_loaded, 'Must load metadata before making the model.'

        if self.is_model_made:
            return

        inputs = self.make_inputs()
        outputs = self.make_model(inputs=inputs,
                                  num_labels=self._metadata[MetaName.NUM_LABELS],
                                  model_mode=model_mode)
        metrics = self.make_metrics()

        loss_weights = self.make_loss_weights()
        loss = self.make_loss()

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self._model = Model(inputs=inputs, outputs=outputs)

        self._model.compile(metrics=metrics,
                            loss=loss,
                            loss_weights=loss_weights,
                            optimizer=optimizer)

        self._is_model_made = True

    def load_metadata(self):
        """
        Extracts the input shape and fits the data scaler
        """
        # Prevent loading metadata twice
        if self.is_metadata_loaded:
            return

        # Normalize the data
        self.dataset.fit_normalizer()
        self.dataset.normalize_data()

        # Save the metadata
        self._metadata[MetaName.INPUT_SHAPE] = tuple(int(x) for x in self.dataset.input_shape)
        self._metadata[MetaName.NUM_LABELS] = int(self.dataset.num_labels)
        self._metadata[MetaName.DATASET_NAME] = self.dataset.dataset_name

        self._is_metadata_loaded = True

    def train(self, save_folder: str, model_mode: ModelMode, should_print: bool):
        """
        Trains the neural network on the given data.
        """
        # Load the metadata for these inputs
        self.load_metadata()

        # Make the model
        self.make(model_mode=model_mode)

        # Make the save folder and get the file name based on the current time
        current_time = datetime.now()
        model_name = '{}_{}'.format(self.name, current_time.strftime('%d-%m-%Y-%H-%M-%S'))

        make_dir(save_folder)
        save_folder = os.path.join(save_folder, self.dataset.dataset_name)
        make_dir(save_folder)

        save_folder = os.path.join(save_folder, current_time.strftime('%d-%m-%Y'))
        make_dir(save_folder)

        save_path = os.path.join(save_folder, '{}.h5'.format(model_name))

        # Make the callbacks in preparation for model fitting
        checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=0, save_best_only=True)
        early_stopping = EarlyStopping(patience=self.hypers[EARLY_STOP_PATIENCE])
        callbacks = [checkpoint, early_stopping]

        # Unpack the dataset
        train_inputs = self.dataset.get_train_inputs()
        train_labels = self.dataset.get_train_labels()

        val_inputs = self.dataset.get_val_inputs()
        val_labels = self.dataset.get_val_labels()

        verbose = 1 if should_print else 2
        history = self._model.fit(train_inputs, 
                                  train_labels,
                                  batch_size=self.hypers[BATCH_SIZE],
                                  epochs=self.hypers[NUM_EPOCHS],
                                  callbacks=callbacks,
                                  validation_data=(val_inputs, val_labels),
                                  verbose=verbose)


        # Save the training history
        train_history_path = os.path.join(save_folder, '{}_train-log.pkl.gz'.format(model_name))

        train_log: Dict[str, List[float]] = dict()
        for key, value in history.history.items():
            train_log[key] = list(map(float, value))

        save_pickle_gz(train_log, train_history_path)

        # Save the hyperparameters and metadata
        model_params = {
            'hypers': self.hypers,
            'metadata': { key.name.upper(): value for key, value in self.metadata.items() }
        }

        model_params_path = os.path.join(save_folder, '{}_model-params.json.gz'.format(model_name))
        save_json_gz(model_params, model_params_path)

    def test(self, should_approx: bool) -> np.ndarray:
        """
        Runs the given operation on the test set.
        """
        test_inputs=self.dataset.get_test_inputs()
        return self.compute_probs(inputs=test_inputs, should_approx=should_approx)

    def validate(self, should_approx: bool) -> np.ndarray:
        """
        Computes the output probabilities on the validation set.
        """
        val_inputs = self.dataset.get_val_inputs()
        return self.compute_probs(inputs=val_inputs, should_approx=should_approx)

    def compute_probs(self, inputs: np.ndarray, should_approx: bool) -> np.ndarray:
        """
        Computes the predicted probabilites on the given dataset.
        """
        preds = self._model.predict(inputs, batch_size=self.hypers[BATCH_SIZE], verbose=0)

        if len(preds) == 1:
            return preds

        probs = np.concatenate([np.expand_dims(arr, axis=1) for arr in preds], axis=1)

        if should_approx:
            logits = np.log(probs + SMALL_NUMBER)
            probs = approx_softmax(logits, axis=-1)

        return probs

    @classmethod
    def restore(cls, path: str, model_mode: ModelMode):
        """
        Restores the model saved at the given path
        """
        # Read in the model parameters
        model_params_path = '{}_model-params.json.gz'.format(path.replace('.h5', ''))
        model_params = read_json_gz(model_params_path)

        hypers = model_params['hypers']
        metadata = model_params['metadata']

        # Load the saved data
        metadata = { MetaName[key.upper()]: value for key, value in metadata.items() }

        # Build the model
        model = cls(hypers=hypers, dataset_name=metadata[MetaName.DATASET_NAME])

        model._metadata = metadata
        model._is_metadata_loaded = True

        model.dataset.fit_normalizer()
        model.dataset.normalize_data()

        model.make(model_mode=model_mode)
        model._model.load_weights(path)

        return model
