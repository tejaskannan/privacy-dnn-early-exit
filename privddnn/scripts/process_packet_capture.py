import csv
from argparse import ArgumentParser
from typing import Any, Dict, List

from privddnn.utils.file_utils import save_jsonl_gz


MCU_IP = '10.150.6.9'
SERVER_IP = '10.150.128.28'
STEP_SIZE = 1
THRESHOLD = 500


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    current_time = 0
    current_length = 0
    current_step = -1

    message_lengths: List[int] = []
    message_times: List[float] = []

    with open(args.csv_path, 'r') as fin:
        reader = csv.reader(fin, delimiter=',', quotechar='"')

        for idx, line in enumerate(reader):
            # Skip the headings
            if idx == 0:
                continue
    
            timestep = float(line[1])
            source_ip = line[2]
            dst_ip = line[3]
            length = int(line[5])

            if source_ip != MCU_IP:
                continue

            if timestep > current_time + STEP_SIZE:
                # Write out the previous information
                if current_step > -1:
                    message_lengths.append(current_length)
                    message_times.append(current_time)

                current_time = timestep
                current_length = 0
                current_step += 1

            current_length += length

        message_lengths.append(current_length)
        message_times.append(current_time)

    results: List[Dict[str, Any]] = []
    for length, time in zip(message_lengths, message_times):
        sample_dict = {
            'exit_decision': int(length > THRESHOLD),
            'length': length,
            'time': time
        }

    save_jsonl_gz(results, args.output_path)
