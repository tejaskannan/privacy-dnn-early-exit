import sys
import time
import os
from argparse import ArgumentParser
from typing import List, Dict

from dnn import DenseNeuralNetwork
from ble_manager import BLEManager
from result_manager import ResultManager
from encryption import decrypt_aes128
from decode import encode_inputs, decode_response, ResponseType
from privddnn.dataset import Dataset
from privddnn.dataset.data_iterators import make_data_iterator, DataIterator


START_BYTE = b'\xAA'
RESET_BYTE = b'\xBB'
SEND_BYTE = b'\xCC'
ACK_BYTE = b'\xDD'
PRECISION = 10

START_RESPONSE = b'\xAB'
RESET_RESPONSE = b'\xCD'

MAC_ADDRESS = 'A0:6C:65:CF:81:D4'
HANDLE = 18
HCI_DEVICE = 'hci0'

PERIOD = 2

# Hard coded symmetric encryption key
AES128_KEY = bytes.fromhex('349fdc00b44d1aaacaa3a2670fd44244')


def run_server(dnn: DenseNeuralNetwork, dataset_iterator: DataIterator, num_samples: int, result_manager: ResultManager, offset: int):
    ble_manager = BLEManager(mac_addr=MAC_ADDRESS,
                             handle=HANDLE,
                             hci_device=HCI_DEVICE)

    # Create a connection
    ble_manager.start()

    # Reset the device
    did_recv = ble_manager.send_and_expect_byte(value=RESET_BYTE,
                                                expected=RESET_RESPONSE)

    if not did_recv:
        print('Could not reset device. Quitting.')
        return

    # Break the connection
    time.sleep(0.25)
    ble_manager.stop()

    print('Press any key to start.')
    x = input()

    # Send the start signal
    sample_count = 0
    for idx, (sample_inputs, _, sample_label) in enumerate(dataset_iterator):
        if (sample_count >= num_samples):
            break
        elif (idx < offset):
            continue

        message = encode_inputs(sample_inputs, precision=PRECISION)

        while not ble_manager.start(max_retries=10):
            print('Timed out. Trying again to connect.')

        did_recv = ble_manager.send_and_expect_byte(value=START_BYTE,
                                                    expected=START_RESPONSE)

        if not did_recv:
            print('Could not send {} inputs. Skipping...')
            continue

        response = ble_manager.query(value=message)

        #time.sleep(0.5)  # Wait for the device to process the information

        #response = ble_manager.query(value=SEND_BYTE)
        #plaintext = decrypt_aes128(ciphertext=response, key=AES128_KEY)

        result = decode_response(response, key=AES128_KEY, precision=PRECISION)
        message_size = len(response)

        if result.response_type == ResponseType.CONTINUE:
            dnn_prediction = dnn(inputs=result.value)
            prediction = dnn_prediction.pred
        else:
            prediction = result.value

        print('Recv Sample {:02d}. Prediction {:02d}'.format(idx, prediction), end='\r')

        result_manager.add_result(pred=int(prediction), message_size=int(message_size), label=int(sample_label))

        time.sleep(0.1)
        ble_manager.stop()

        # Pause for half the period. We try connecting again early to 
        # connect quickly once the BT module wakes up.
        time.sleep(PERIOD / 2)
        sample_count += 1

    print()
    ble_manager.stop()


if __name__ == '__main__':
    parser = ArgumentParser('Server module for interfacing with the MSP430 executing distributed DNNs.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved model parameters.')
    parser.add_argument('--dataset-name', type=str, required=True, help='Name of the dataset.')
    parser.add_argument('--dataset-order', type=str, required=True, help='Name of the dataset order.', choices=['same-label', 'nearest-neighbor'])
    parser.add_argument('--window-size', type=int, required=True, help='The ordering window size.')
    parser.add_argument('--num-samples', type=int, required=True, help='The number of samples to evaluate on.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the output jsonl.gz file.')
    parser.add_argument('--data-fold', type=str, default='test', choices=['train', 'val', 'test'], help='The dataset split to use.')
    parser.add_argument('--offset', type=int, help='The sample index to start at.')
    args = parser.parse_args()

    if os.path.exists(args.output_path):
        print('You are going to overwrite the file {}. Confirm [Y/N]:'.format(args.output_path), end=' ')
        response = input()

        if response.lower() not in ('y', 'yes'):
            print('Quitting.')
            sys.exit(0)

        os.remove(args.output_path)

    dataset = Dataset(args.dataset_name)
    dataset_iterator = make_data_iterator(name=args.dataset_order,
                                          dataset=dataset,
                                          pred_probs=None,
                                          window_size=args.window_size,
                                          num_reps=1,
                                          fold=args.data_fold)

    dnn = DenseNeuralNetwork(args.model_path)
    result_manager = ResultManager(args.output_path)

    will_destroy = (args.offset == 0) and (result_manager.does_exist)

    if will_destroy:
        print('You are going to overwrite the file {}. Confirm [Y/N]:'.format(result_manager.path), end=' ')
        response = input()

        if response.lower() not in ('y', 'yes'):
            print('Quitting.')
            sys.exit(0)

    result_manager.reset(should_destroy=will_destroy)

    print('Starting Server...')
    run_server(dataset_iterator=dataset_iterator,
               dnn=dnn,
               result_manager=result_manager,
               num_samples=args.num_samples,
               offset=args.offset)
