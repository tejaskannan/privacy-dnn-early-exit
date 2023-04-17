import sys
import time
import os
from argparse import ArgumentParser
from subprocess import CalledProcessError
from typing import List, Dict

from dnn import DenseNeuralNetwork
from ble_manager import BLEManager
from result_manager import ResultManager
from encryption import decrypt_aes128, AES_BLOCK_SIZE
from decode import encode_inputs, decode_response, ResponseType
from privddnn.dataset import Dataset
from privddnn.dataset.data_iterators import make_data_iterator, DataIterator


INIT_BYTE = b'\xEE'
SEND_BYTE = b'\xCC'
ACK_BYTE = b'\xDD'
INIT_RESPONSE = b'\xEF'

MAC_ADDRESS = 'A0:6C:65:CF:81:D4'
HANDLE = 18
HCI_DEVICE = 'hci0'

PERIOD = 1


def run_server(num_samples: int):
    ble_manager = BLEManager(mac_addr=MAC_ADDRESS,
                             handle=HANDLE,
                             hci_device=HCI_DEVICE)

    # Create a connection
    ble_manager.start()

    # Reset the device
    did_recv = ble_manager.send_and_expect_byte(value=INIT_BYTE,
                                                expected=INIT_RESPONSE)

    if not did_recv:
        print('Could not initialize device. Quitting.')
        return

    # Break the connection
    ble_manager.send(value=ACK_BYTE)
    time.sleep(0.1)
    ble_manager.stop()

    time.sleep(PERIOD / 2)

    # Send the start signal
    for idx in range(num_samples):
        while not ble_manager.start(max_retries=10):
            print('Timed out. Trying again to connect.')

        response = ble_manager.query(value=SEND_BYTE)

        print('Sample {}, Rec. {} bytes'.format(idx + 1, len(response)), end='\r')

        ble_manager.send(value=ACK_BYTE)

        # Pause for half the period. We try connecting again early to 
        # connect quickly once the BT module wakes up.
        time.sleep(0.1)
        ble_manager.stop()
        time.sleep(PERIOD / 2)

    print()
    ble_manager.stop()


if __name__ == '__main__':
    parser = ArgumentParser('Server module for measuring the energy consumption of distributed DNNs.')
    parser.add_argument('--num-samples', type=int, required=True, help='The number of samples to evaluate on.')
    args = parser.parse_args()

    print('Starting Server...')
    run_server(num_samples=args.num_samples)
