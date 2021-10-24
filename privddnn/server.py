import socket
import numpy as np
from argparse import ArgumentParser
from typing import List

from privddnn.classifier import BaseClassifier, ModelMode
from privddnn.exiting.early_exit import make_policy, EarlyExiter, ExitStrategy
from privddnn.utils.encoding import decode, encode_prediction, FLOAT_SIZE
from privddnn.utils.encryption import encrypt_aes128, decrypt_aes128, round_to_block, AES_BLOCK_SIZE
from privddnn.restore import restore_classifier


class Server:

    def __init__(self, host: str, port: int, clf: BaseClassifier, exit_policy: EarlyExiter):
        self._host = host
        self._port = port
        self._clf = clf
        self._exit_policy = exit_policy
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self._aes_key = bytes.fromhex('b5c723a7b34c45a185011b6d2d23f18c')

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def classifier(self) -> BaseClassifier:
        return self._clf

    @property
    def exit_policy(self) -> EarlyExiter:
        return self._exit_policy

    def __enter__(self):
        self._socket.bind((self.host, self.port))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._socket.close()

    def run(self, num_samples: int, input_size: int):
        """
        Starts the inference server.

        Args:
            num_samples: The number of samples in the experiment
            input_size: The size of each input (in bytes)
        Returns:
            Nothing. The predictions are sent back to the client.
        """
        # Listen and Accept any inbound connections
        self._socket.listen()
        conn, addr = self._socket.accept()

        # Round the input size to the nearest block multiple and
        # account for the IV
        input_size = AES_BLOCK_SIZE + round_to_block(input_size, block_size=AES_BLOCK_SIZE)

        print('Accepted Connection from {}'.format(addr))

        with conn:
            for idx in range(num_samples):
                # Receive the input sample
                recv = conn.recv(input_size)
                decrypted = decrypt_aes128(ciphertext=recv,
                                           key=self._aes_key)

                # Parse the message into the proper data type
                inputs = decode(encoded=decrypted, dtype='float')

                # Execute the inference on the first output
                probs = self.classifier.predict_sample(inputs=inputs, level=0)
                level = self.exit_policy.get_level(probs=probs)

                # If needed, elevate the prediction to the second output
                if level == 1:
                    probs = self.classifier.predict_sample(inputs=inputs, level=1)

                # Get the model's prediction
                pred = int(np.argmax(probs))

                # Send the prediction back to the client
                encoded_pred = encode_prediction(pred)
                encrypted_pred = encrypt_aes128(message=encoded_pred,
                                                key=self._aes_key)

                conn.send(encrypted_pred)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--exit-rate', type=float, required=True)
    parser.add_argument('--max-num-samples', type=int)
    parser.add_argument('--port', type=int, default=5600)
    args = parser.parse_args()

    assert args.exit_rate >= 0.0 and args.exit_rate <= 1.0, 'Exit rate must be in [0, 1]'

    clf: BaseClassifier = restore_classifier(model_path=args.model_path, model_mode=ModelMode.TEST)

    # Get the data size
    inputs = clf.dataset.get_train_inputs()
    input_size = np.prod(inputs.shape) * FLOAT_SIZE

    # Get the number of input samples
    num_samples = args.max_num_samples if args.max_num_samples is not None else len(clf.dataset.get_test_labels())

    # Make the exit policy
    exit_policy = make_policy(strategy=ExitStrategy[args.policy.upper()],
                              rates=[args.exit_rate, 1.0 - args.exit_rate],
                              model_path=args.model_path)

    with Server(host='localhost', port=args.port, clf=clf, exit_policy=exit_policy) as server:
        print('Started Server.')
        server.run(num_samples=num_samples, input_size=input_size)
