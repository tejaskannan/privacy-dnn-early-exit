import os.path
from typing import List, Dict

from privddnn.utils.file_utils import append_jsonl_gz


class ResultManager:

    def __init__(self, path: str):
        self._path = path
        self._results: List[Dict[str, int]] = []

    @property
    def path(self) -> str:
        return self._path

    @property
    def does_exist(self) -> bool:
        return os.path.exists(self._path)

    @property
    def results(self) -> List[Dict[str, int]]:
        return self._results

    def reset(self, should_destroy: bool):
        if self.does_exist and should_destroy:
            os.remove(self._path)

        self._results: List[Dict[str, int]] = []

    def add_result(self, pred: int, message_size: int, label: int):
        record = dict(pred=pred, message_size=message_size, label=label)
        self._results.append(record)
        append_jsonl_gz(record, self._path)
