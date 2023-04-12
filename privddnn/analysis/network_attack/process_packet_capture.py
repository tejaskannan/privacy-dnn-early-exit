import csv
import numpy as np
from argparse import ArgumentParser
from typing import Any, Dict, List

from privddnn.utils.file_utils import save_jsonl_gz


PERIOD = 0.1
RETRANSMIT_TIME = 0.15
SIZE_THRESHOLD = 500


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv-path', type=str, required=True)
    parser.add_argument('--server-ip', type=str, required=True)
    parser.add_argument('--client-ip', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    current_time = 0
    current_length = 0
    current_step = -1
    should_skip_next_ack = False

    message_lengths: List[int] = []
    message_times: List[float] = []
    retransmit_times: List[float] = []

    did_retransmit = False
    retransmit_time = 0.0
    prev_packet_idx = -1
    packet_idx = 0

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
            info = line[6]

            if (source_ip not in (args.server_ip, args.client_ip)) or (dst_ip not in (args.server_ip, args.client_ip)):
                continue

            if (source_ip == args.server_ip) and (dst_ip == args.client_ip):
                if 'FIN' in info:
                    should_skip_next_ack = True
            else:
                if 'FIN' in info:
                    should_skip_next_ack = True
                elif ('TCP Spurious Retransmission' in info):
                    retransmit_times.append(timestep)
                    did_retransmit = True
                    retransmit_time = timestep
                elif (('ACK' not in info) or (not should_skip_next_ack)) and ('Retransmission' not in info) and ('SYN' not in info) or ('Out-Of-Order' in info):
                    if did_retransmit and (timestep - retransmit_time) > RETRANSMIT_TIME:
                        did_retransmit = False

                    if did_retransmit:
                        if (packet_idx - prev_packet_idx) > 1:
                            message_lengths.append(current_length)
                            message_times.append(current_time)

                            current_time = timestep
                            current_length = 0
                            current_step += 1

                        current_length += length
                    else:
                        if (timestep > current_time + PERIOD):
                            # Write out the previous information
                            if current_step > -1:
                                message_lengths.append(current_length)
                                message_times.append(current_time)

                            current_time = timestep
                            current_length = 0
                            current_step += 1

                        current_length += length
                        did_retransmit = False

                    prev_packet_idx = packet_idx

            if ('Out-Of-Order' not in info) and ('TCP Retransmission' not in info) and ('Dup' not in info) and ('Previous Segment' not in info):
                packet_idx += 1

        if current_length > 0:
            message_lengths.append(current_length)
            message_times.append(current_time)

    # Guess the average period using the median difference
    differences = np.subtract(message_times[1:], message_times[:-1])
    period = np.median(differences)
    iqr = np.percentile(differences, 75) - np.percentile(differences, 25)
    cutoff = period + 4 * iqr

    # Get the median packet length
    avg_length = np.average(message_lengths)
    median_length = np.median(message_lengths)
    length_iqr = np.percentile(message_lengths, 75) - np.percentile(differences, 25)
    length_cutoff = median_length + 2.5 * length_iqr
    prev_time = None

    results: List[Dict[str, Any]] = []
    for length, time in zip(message_lengths, message_times):
        sample_dict = {
            'exit_decision': int(length > SIZE_THRESHOLD),
            'length': length,
            'time': time
        }
        results.append(sample_dict)

        if (prev_time is not None) and (time - prev_time) > cutoff:
            # Get the re-transmissions between the two times
            retransmits = list(filter(lambda t: (t >= prev_time) and (t < time), retransmit_times))

            if len(retransmits) == 0:
                t = prev_time + period
                while t < (time - period):
                    sample_dict = {
                        'exit_decision': 0,
                        'length': 0,
                        'time': t
                    }
                    results.append(sample_dict)

                    t += period

        prev_time = time

    print('Saving {} results.'.format(len(results)))

    save_jsonl_gz(results, args.output_path)
