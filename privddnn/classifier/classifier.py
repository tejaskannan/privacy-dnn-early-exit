import numpy as np
from privddnn.dataset.dataset import Dataset
from .constants import OpName, ModelMode


class BaseClassifier:

    def __init__(self, dataset_name: str):
        self._dataset = Dataset(dataset_name=dataset_name)

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def name(self) -> str:
        raise NotImplementedError()

    def validate(self, opName: OpName) -> np.ndarray:
        raise NotImplementedError()

    def test(self, opName: OpName) -> np.ndarray:
        raise NotImplementedError()

    def save(path: str):
        raise NotImplementedError()

    @classmethod
    def restore(cls, path: str, model_mode: ModelMode):
        raise NotImplementedError()
