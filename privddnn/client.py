import numpy as np
import socket
import time
from argparse import ArgumentParser
from collections import namedtuple
from typing import Optional

from privddnn.dataset.dataset import Dataset
from privddnn.utils.encoding import encode, decode_prediction, INT_SIZE
from privddnn.utils.encryption import encrypt_aes128, decrypt_aes128, round_to_block, AES_BLOCK_SIZE


EvalResult = namedtuple('EvalResult', ['predictions', 'latency', 'labels'])


class Client:

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self._aes_key = bytes.fromhex('b5c723a7b34c45a185011b6d2d23f18c')

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def __enter__(self):
        self._socket.connect((self.host, self.port))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._socket.close()

    def run(self, inputs: np.ndarray, labels: np.ndarray, max_num_samples: Optional[int]) -> EvalResult:
        """
        Executes the client process which queries the server for predictions.

        Args:
            inputs: An array of input features
            labels: An array of data labels for each input sample
            max_num_samples: An optional maximum number of input samples
        """
        assert len(inputs) == len(labels), 'Misaligned input and label arrays'

        num_samples = max_num_samples if max_num_samples is not None else len(labels)
        num_samples = min(len(labels), num_samples)

        # Lists to record the evaluation results
        predictions: List[int] = []
        data_labels: List[int] = []
        latency: List[float] = []

        for idx in range(num_samples):
            # Encode the input features
            input_features = inputs[idx]
            encoded_message = encode(array=input_features, dtype='float')
            encrypted_message = encrypt_aes128(message=encoded_message,
                                               key=self._aes_key)

            # Send the input data
            start = time.time()
            self._socket.sendall(encrypted_message)

            # Get the response message which encodes the prediction
            response = self._socket.recv(round_to_block(INT_SIZE, block_size=AES_BLOCK_SIZE) + AES_BLOCK_SIZE)
            
            # Record the elapsed time
            end = time.time()
            elapsed = end - start

            # Decrypt the result to extract the prediction
            decrypted_response = decrypt_aes128(ciphertext=response,
                                                key=self._aes_key)

            pred = decode_prediction(decrypted_response)

            # Record the results for this sample
            predictions.append(pred)
            data_labels.append(labels[idx])
            latency.append(elapsed)

        return EvalResult(predictions=predictions, labels=data_labels, latency=latency)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--max-num-samples', type=int)
    parser.add_argument('--port', type=int, default=5600)
    args = parser.parse_args()

    # Normalize the data and get the testing fold
    dataset = Dataset(args.dataset_name)
    dataset.fit_normalizer(is_global=False)
    dataset.normalize_data()

    test_inputs, test_labels = dataset.get_test_inputs(), dataset.get_test_labels()

    with Client(host='localhost', port=args.port) as client:
        eval_result = client.run(inputs=test_inputs,
                                 labels=test_labels,
                                 max_num_samples=args.max_num_samples)

    print(eval_result)
