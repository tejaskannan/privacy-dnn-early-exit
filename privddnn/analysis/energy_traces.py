import numpy as np
import matplotlib.pyplot as plt
import csv
from argparse import ArgumentParser
from typing import Tuple, List, Optional

from privddnn.utils.constants import SMALL_NUMBER


FACTOR = 1.02


def read_energy_trace(path: str) -> Tuple[List[float], List[float], List[float]]:
    time_list: List[float] = []
    power_list: List[float] = []
    energy_list: List[float] = []

    prev_energy = 0.0
    prev_time = 0.0

    with open(path, 'r') as fin:
        reader = csv.reader(fin, delimiter=',', quotechar='"')

        for idx, line in enumerate(reader):
            # Skip the headers
            if idx == 0:
                continue

            t = float(line[0]) * 1e-6   # Time in milliseconds
            current = float(line[1]) * 1e-9  # Current in Amps
            voltage = float(line[2])  # Voltage in mV
            power = current * voltage
            energy = float(line[3]) * 1e-3  # Energy in mJ

            if (t > 250):
                time_list.append(t)
                power_list.append(power)
                energy_list.append(energy)

    return time_list, power_list, energy_list


def get_energy_per_op(time_list: List[float], power_list: List[float], energy_list: List[float], num_ops: int) -> Tuple[List[float], List[float], List[float]]:
    min_power = min(power_list)
    threshold_power = FACTOR * min_power

    start_times: List[float] = []
    end_times: List[float] = []
    op_energy_list: List[float] = []

    start_energy = None

    for time, power, energy in zip(time_list, power_list, energy_list):
        if (start_energy is None) and (power >= threshold_power):
            start_energy = energy
            start_times.append(time)
        elif (start_energy is not None) and (power < threshold_power):
            op_energy = energy - start_energy

            op_energy_list.append(op_energy)
            end_times.append(time)

            start_energy = None

    sorted_idx = np.argsort(op_energy_list)[::-1]
    clipped_idx = sorted_idx[0:num_ops]

    start_times = [start_times[i] for i in clipped_idx]
    end_times = [end_times[i] for i in clipped_idx]
    op_energy_list = [op_energy_list[i] for i in clipped_idx]

    return start_times, end_times, op_energy_list


def plot(time_list: List[float], power_list: List[float], start_times: List[float], end_times: List[float], output_path: Optional[str]):
    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()
        
        # Sample the result down for quicker plotting
        xs = [t for i, t in enumerate(time_list) if i % 10 == 0]
        ys = [p for i, p in enumerate(power_list) if i % 10 == 0]

        ax.plot(xs, ys)

        #min_power = min(power_list)
        #ax.axhline(min_power, color='k')

        #for start_time in start_times:
        #    ax.axvline(start_time, color='orange')

        #for end_time in end_times:
        #    ax.axvline(end_time, color='red')

        ax.set_ylim([0.5, 1.2])

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Power (mV)')
        ax.set_title('MSP430 Device Power for AdaBoost Ensemble Inference')

        if output_path is None:
            plt.show()
        else:
            plt.savefig(output_path, bbox_inches='tight', transparent=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--log-file', type=str, required=True, help='Path to the raw trace CSV file.')
    parser.add_argument('--output-file', type=str, help='Optional path to an output file for plot saving.')
    args = parser.parse_args()

    time_list, power_list, energy_list = read_energy_trace(path=args.log_file)

    start_times, end_times, op_energy = get_energy_per_op(time_list, power_list, energy_list=energy_list, num_ops=8)

    print('Average Energy / Op (mJ): {:6f}'.format(np.average(op_energy)))

    plot(time_list=time_list, power_list=power_list, start_times=start_times, end_times=end_times, output_path=args.output_file)
