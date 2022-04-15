import socket
import json
import os
import sys
import numpy as np
from argparse import ArgumentParser
from Cryptodome.Random import get_random_bytes
from Cryptodome.Cipher import ChaCha20

from privddnn.jetson_nano_dnns.branchynet_dnn import JetsonNanoBranchyNetDNN
from privddnn.utils.file_utils import append_jsonl_gz


PACKET_SIZE = 1024
INT_SIZE = 4
ENCRYPTION_KEY = bytes.fromhex('434e8fad2818f1fe02b48593584a3da424adf501decbe6b470542317e5561210')
NONCE_LEN = 12


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--host', type=str, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--max-num-samples', type=int, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    dnn = JetsonNanoBranchyNetDNN(model_path=args.model_path,
                                  input_shape=(300, ),
                                  num_labels=12)

    if os.path.exists(args.output_path):
        print('The file {} exists and will be overwritten. Continue? [y/n]'.format(args.output_path))
        response = input()

        if response.lower() not in ('y', 'yes'):
            sys.exit(0)

        os.remove(args.output_path)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Set up the socket
        sock.bind((args.host, args.port))
        sock.listen()

        print('Started Server.')

        conn, addr = sock.accept()
        with conn:
            print('Connected by {}'.format(addr))

            message_buffer = bytearray()

            for sample_idx in range(args.max_num_samples):
                # Wait for the client to send over the result
                data = conn.recv(PACKET_SIZE)
                message_buffer.extend(data)

                # Extract the true data length
                length_bytes = int.from_bytes(message_buffer[0:INT_SIZE], byteorder='big')

                while len(message_buffer) < length_bytes:
                    data = conn.recv(PACKET_SIZE)
                    message_buffer.extend(data)

                # Decode the result
                try:
                    message = message_buffer[INT_SIZE:length_bytes]
                    nonce = message[0:NONCE_LEN]
                    ciphertext = message[NONCE_LEN:]

                    cipher = ChaCha20.new(key=ENCRYPTION_KEY, nonce=nonce)
                    plaintext = cipher.decrypt(ciphertext).decode('utf-8')

                    sample_dict = json.loads(plaintext)
                except json.decoder.JSONDecodeError as ex:
                    print(message)
                    print('Reported Length: {}, Total Length: {}, Message length: {}'.format(length_bytes, len(message_buffer), len(message)))
                    raise ex

                # If system did not early exit, use the full model to compute the prediction
                exit_decision = sample_dict['exit_decision']
                label = sample_dict['label']
                message_size = length_bytes

                if exit_decision == 1:
                    input_features = np.expand_dims(sample_dict['input_features'], axis=0)  # [1, D]
                    hidden_features = np.expand_dims(sample_dict['model_state'], axis=0)  # [1, K]

                    _, full_probs = dnn.execute_full_model(inputs=input_features,
                                                           model0_state=hidden_features)

                    full_pred = np.argmax(full_probs[0])
                    result_dict = {
                        'prediction': int(full_pred)
                    }
                else:
                    result_dict = {
                        'prediction': sample_dict['prediction']
                    }

                result_dict['exit_decision'] = exit_decision
                result_dict['message_size'] = message_size
                result_dict['label'] = label

                append_jsonl_gz(result_dict, args.output_path)
                message_buffer = message_buffer[length_bytes:]

                print('Completed {} samples.'.format(sample_idx + 1), end='\r')
            print()
