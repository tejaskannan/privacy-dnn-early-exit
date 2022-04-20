import time
from argparse import ArgumentParser
from typing import List

from ble_manager import BLEManager
from encryption import decrypt_aes128
from privddnn.dataset import Dataset
from privddnn.dataset.data_iterators import make_data_iterator


START_BYTE = b'\xAA'
RESET_BYTE = b'\xBB'
SEND_BYTE = b'\xCC'
ACK_BYTE = b'\xDD'

START_RESPONSE = b'\xAB'
RESET_RESPONE = b'\xCD'

MAC_ADDRESS = '00:35:FF:13:A3:1E'
HANDLE = 18
HCI_DEVICE = 'hci0'

PERIOD = 1

# Hard coded symmetric encryption key
AES128_KEY = bytes.fromhex('349fdc00b44d1aaacaa3a2670fd44244')


def run_server(num_samples: int, labels: List[int], output_path: str):
    ble_manager = BLEManager(mac_addr=MAC_ADDRESS,
                             handle=HANDLE,
                             hci_device=HCI_DEVICE)

    # Reset the device
    did_recv = ble_manager.send_and_expect_byte(value=RESET_BYTE,
                                                expected=RESET_RESPONSE)

    if not did_recv:
        print('Could not reset device. Quitting.')
        return

    print('Press any key to start.')
    x = input()

    did_recv = ble_manager.send_and_expect_byte(value=START_BYTE,
                                                expected=START_RESPONSE)

    if not did_recv:
        print('Could not start experiment. Quitting.')
        return

    ble_manager.send(value=ACK_BYTE)

    for idx in range(num_samples):
        while not ble_manager.start(max_retries=10):
            print('Timed out. Trying again to connect.')

        response = ble_manager.query(value=SEND_BYTE)
        plaintext = decrypt_aes128(ciphertext=response, key=AES128_KEY)

        prediction = int.from_bytes(plaintext[0], byteorder='big')
        exit_decision = int.from_bytes(plaintext[1], byteorder='big')

        ble_manager.send(value=ACK_BYTE)

        record = dict(prediction=prediction, exit_decision=exit_decision, label=labels[idx])
        append_jsonl_gz(record, output_path)

        time.sleep(PERIOD / 2.0)   # Pause for half the period

    ble_manager.stop()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--dataset-order', type=str, required=True)
    parser.add_argument('--window-size', type=int, required=True)
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--data-fold', type=str, default='test', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    if os.path.exists(args.output_file):
        print('You are going to overwrite the file {}. Confirm [Y/N]:'.format(args.output_file), end=' ')
        response = input()

        if response.lower() not in ('y', 'yes'):
            print('Quitting.')
            sys.exit(0)

        os.remove(args.output_file)

    dataset = Dataset(args.dataset_name)
    dataset_iterator = make_data_iterator(name=args.dataset_order,
                                          dataset=dataset,
                                          pred_probs=None,
                                          window_size=args.window_size,
                                          num_reps=1,
                                          fold=args.fold)

    labels: List[int] = []
    for idx, (_, _, label) in enumerate(dataset_iterator):
        if idx >= args.num_samples:
            break

        labels.append(label)

    print('Starting Server...')
    run_server(labels=labels,
               num_samples=args.num_samples,
               output_path=args.output_path)
