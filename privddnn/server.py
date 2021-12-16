import numpy as np
import time
import os
import sys
from argparse import ArgumentParser
from collections import namedtuple, Counter
from typing import Optional, Iterable, List

from privddnn.classifier import ModelMode
from privddnn.dataset import Dataset
from privddnn.dataset.data_iterators import DataIterator, make_data_iterator
from privddnn.device.ble_manager import BLEManager
from privddnn.device.encryption import AES_BLOCK_SIZE, decrypt_aes128
from privddnn.device.decode import decode_exit_message, decode_elevate_message, MessageType, get_message_type, decode_buffered_message
from privddnn.device.dnn import DenseNeuralNetwork
from privddnn.exiting.early_exit import ExitStrategy
from privddnn.restore import restore_classifier
from privddnn.utils.file_utils import read_pickle_gz, save_json_gz, make_dir


MAC_ADDRESS = '00:35:FF:13:A3:1E'
BLE_HANDLE = 18
HCI_DEVICE = 'hci0'
PERIOD = 2
LENGTH_SIZE = 2

# Special bytes to handle sensor operation
START_BYTE = b'\xAA'
RESET_BYTE = b'\xBB'
QUERY_BYTE = b'\xCC'

START_RESPONSE = b'\xAB'
RESET_RESPONSE = b'\xCD'

# Hard coded symmetric encryption key
AES128_KEY = bytes.fromhex('349fdc00b44d1aaacaa3a2670fd44244')


def execute(model_path: str,
            data_iterator: DataIterator,
            precision: int,
            output_file: str,
            policy_type: ExitStrategy,
            window_size: Optional[int],
            max_num_samples: Optional[int]):
    """
    Starts the device client. This function either sends data and expects the device to respond with predictions
    or assumes that the device performs sensing on its own.

    Args:
        model_path: Path to the serialized dense neural network.
        labels: The true data labels. Used only for accuracy computation.
        precision: The precision of each data measurement.
        output_file: The path to the output json gz file.
        max_num_samples: Maximum number of sequences before terminating the experiment.
    """
    # Initialize the device manager
    device_manager = BLEManager(mac_addr=MAC_ADDRESS, handle=BLE_HANDLE, hci_device=HCI_DEVICE)

    # Lists to store experiment results
    num_bytes_list: List[int] = []
    labels_list: List[int] = []
    predictions_list: List[int] = []

    num_correct = 0
    total_samples = 0

    rand = np.random.RandomState(seed=54121)

    # Create the top-level DNN
    model_weights = read_pickle_gz(model_path)['weights']
    dnn_model = DenseNeuralNetwork(model_weights)

    print('==========')
    print('Starting experiment')
    print('==========')

    # Start the device
    try:
        device_manager.start()
        reset_result = device_manager.send_and_expect_byte(value=RESET_BYTE, expected=RESET_RESPONSE)

        if (reset_result == False):
            print('Could not reset the device.')
            return
    finally:
        device_manager.stop()

    print('Tested Connection. Press any key to start...')
    x = input()

    try:
        # Send the 'start' sequence
        device_manager.start()

        start_response = device_manager.send_and_expect_byte(value=START_BYTE, expected=START_RESPONSE)

        if (start_response == False):
            print('Could not start device.')
            return

        device_manager.stop()

        print('Sent Start signal.')

        for idx, (_, _, label) in enumerate(data_iterator):
            if (max_num_samples is not None) and (idx >= max_num_samples):
                break

            # Delay to align with sampling period
            time.sleep(PERIOD)

            if policy_type == ExitStrategy.BUFFERED_MAX_PROB:
                labels_list.append(int(label))
                total_samples += 1

                if (total_samples % window_size) != 0:
                    continue

                # Send the fetch character and wait for the response
                did_connect = device_manager.start(wait_time=0.1)

                if not did_connect:
                    print('[WARNING] Could not connect.')
                    continue
 
                response = device_manager.query(value=QUERY_BYTE)
                device_manager.stop()

                message_byte_count = len(response)

                if message_byte_count == 0:
                    print('[WARNING] Recieved message with 0 bytes.')
                    continue

                # Decrypt the response
                try:
                    data = decrypt_aes128(response, AES128_KEY)
                except ValueError as ex:
                    print('[WARNING] Could not decrypt response due to {}.'.format(ex))
                    continue

                # Extract the length
                length = int.from_bytes(data[0:LENGTH_SIZE], 'big')

                # Clip the result to the true length and remove any padding characters
                data = data[LENGTH_SIZE:length + LENGTH_SIZE]

                message_type = get_message_type(data)
                assert message_type == MessageType.BUFFERED, 'Message must be of the `buffered` type.'

                buffered_result = decode_buffered_message(data, precision=precision)

                print(data)
                print(buffered_result.inputs[0])
                print(buffered_result.hidden[0])

                num_bytes_list.append(int(message_byte_count))

                for window_idx in range(window_size):
                    if window_idx in buffered_result.elevate_indices:
                        elev_idx = buffered_result.elevate_indices.index(window_idx)
                        elev_inputs = np.array(buffered_result.inputs[elev_idx] + buffered_result.hidden[elev_idx]).reshape(-1, 1)
                        logits = dnn_model(elev_inputs)
                        pred = np.argmax(logits[:, 0])
                    else:
                        pred = buffered_result.preds[window_idx]

                    predictions_list.append(int(pred))

                    label = labels_list[-window_size + window_idx]
                    num_correct += (pred == label)

                # Save the results so far
                results_dict = {
                    'accuracy': (num_correct / total_samples),
                    'preds': predictions_list,
                    'labels': labels_list,
                    'num_bytes': num_bytes_list
                }
                save_json_gz(results_dict, output_file)

                print('Completed {} Sequences. Accuracy so far: {:.6f}'.format(total_samples, num_correct / total_samples), end='\r')
            else:
                # Send the fetch character and wait for the response
                did_connect = device_manager.start(wait_time=0.1)

                if not did_connect:
                    print('[WARNING] Could not connect.')
                    continue

                response = device_manager.query(value=QUERY_BYTE)
                device_manager.stop()

                message_byte_count = len(response)
                if message_byte_count == 0:
                    print('[WARNING] Recieved message with 0 bytes.')
                    continue

                # Decrypt the response
                try:
                    data = decrypt_aes128(response, AES128_KEY)
                except ValueError as ex:
                    print('[WARNING] Could not decrypt response due to {}.'.format(ex))
                    continue

                # Extract the length
                length = int.from_bytes(data[0:LENGTH_SIZE], 'big')

                # Clip the result to the true length and remove any padding characters
                data = data[LENGTH_SIZE:length + LENGTH_SIZE]

                # Decode the result and get the prediction based on the exit type.
                message_type = get_message_type(data)

                if message_type == MessageType.EXIT:
                    pred = decode_exit_message(data)
                elif message_type == MessageType.ELEVATE:
                    elevate_result = decode_elevate_message(data)

                    elev_inputs = np.array(elevate_result.inputs + elevate_result.hidden).reshape(-1, 1)
                    logits = dnn_model(elev_inputs)
                    pred = np.argmax(logits[:, 0])
                else:
                    raise ValueError('Unsupported message type {}'.format(message_type))

                # Log the results so far
                predictions_list.append(pred)
                labels_list.append(label)
                num_bytes_list.append(message_byte_count)

                num_correct += int(pred == label)
                total_samples += 1

                # Save the results so far
                results_dict = {
                    'accuracy': (num_correct / total_samples),
                    'preds': predictions_list,
                    'labels': labels_list,
                    'num_bytes': num_bytes_list
                }
                save_json_gz(results_dict, output_file)

                print('Completed {} Sequences. Accuracy so far: {:.6f}'.format(idx + 1, num_correct / total_samples), end='\r')

        print()
    finally:
        device_manager.stop()

    print('Completed. Accuracy: {:.6f}'.format(num_correct / total_samples))

    # Save the final results
    results_dict = {
        'accuracy': (num_correct / total_samples),
        'preds': predictions_list,
        'labels': labels_list,
        'num_bytes': num_bytes_list
    }
    save_json_gz(results_dict, output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the serialized Branchynet DNN model weights.')
    parser.add_argument('--dataset-order', type=str, required=True, help='The name of the dataset ordering.')
    parser.add_argument('--exit-policy', type=str, required=True, help='Name of the exit policy.')
    parser.add_argument('--exit-rate', type=float, required=True, help='The target early exit rate (for logging purposes).')
    parser.add_argument('--precision', type=int, required=True, help='The fixed point precision of measurement values. Must match the MSP430 implementation.')
    parser.add_argument('--output-folder', type=str, required=True, help='The folder in which to save results.')
    parser.add_argument('--dataset-window-size', type=int, help='The window size for nearest-neighbor dataset orderings.')
    parser.add_argument('--policy-window-size', type=int, help='The window size for buffered exit policies.')
    parser.add_argument('--max-samples', type=int, help='An optional maximum number of samples.')
    args = parser.parse_args()

    assert (args.precision > 0) and (args.precision < 16), 'The precision must be in the range (0, 16)'
    assert (args.exit_rate >= 0.0) and (args.exit_rate <= 1.0), 'The exit rate must be in [0, 1]'

    # Extract the model name
    file_name = os.path.basename(args.model_path)
    model_name = file_name.replace('.pkl.gz', '')

    make_dir(args.output_folder)
    output_file = os.path.join(args.output_folder, '{}_{}_{}.json.gz'.format(model_name, args.exit_policy, int(round(args.exit_rate * 100))))

    if os.path.exists(output_file):
        print('The output file {} exists. Do you want to overwrite it? [Y/N]'.format(output_file))
        d = input()
        if d.lower() not in ('y', 'yes'):
            sys.exit(0)

    # Restore the model
    model = restore_classifier(args.model_path, model_mode=ModelMode.TEST)

    # Make the dataset iterator
    data_iterator = make_data_iterator(name=args.dataset_order,
                                       dataset=model.dataset,
                                       clf=None,
                                       window_size=args.dataset_window_size,
                                       num_trials=1,
                                       fold='val')

    exit_strategy = ExitStrategy[args.exit_policy.upper()]

    # Run the server
    execute(model_path=args.model_path,
            data_iterator=data_iterator,
            precision=10,
            max_num_samples=args.max_samples,
            policy_type=exit_strategy,
            window_size=args.policy_window_size,
            output_file=output_file)
