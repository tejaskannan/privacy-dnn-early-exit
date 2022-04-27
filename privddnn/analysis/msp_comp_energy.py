import csv
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.integrate import trapz
from typing import Tuple, List, Optional

from privddnn.utils.constants import SMALL_NUMBER


THRESHOLD = 0.5
TIME_DELTA = 1


def filter_windows(start_idx: List[int], end_idx: List[int], time_list: List[float], power_list: List[float]) -> Tuple[List[int], List[int]]:
    filtered_start_idx: List[int] = []
    filtered_end_idx: List[int] = []

    for start, end in zip(start_idx, end_idx):
        time_delta = time_list[end] - time_list[start]

        if time_delta > TIME_DELTA:
            filtered_start_idx.append(start)
            filtered_end_idx.append(end)

    # Walk the start times up until a minimum and walk the end times back until a minimum (or constant)
    adjusted_start_idx: List[int] = []
    adjusted_end_idx: List[int] = []

    for start, end in zip(filtered_start_idx, filtered_end_idx):

        while (power_list[start] > power_list[start + 1]) or (power_list[start] > 0.27):
            start += 1

        while (abs(power_list[start] - power_list[start + 1]) < 0.01):
            start += 1

        while (power_list[end] > power_list[end - 1]):
            end -= 1

        adjusted_start_idx.append(start)
        adjusted_end_idx.append(end)

    return adjusted_start_idx, adjusted_end_idx


def get_energy_per_period(path: str, output_file: Optional[str]) -> List[float]:
    """
    Computes the energy from the given EnergyTrace CSV file.
    
    Args:
        path: Path to the EnergyTrace CSV file
        output_file: Optional path to output file for the plotted result
    Returns:
        A tuple with 2 elements:
            (1) The energy in mJ
            (2) The empirical power in mW
    """
    time_list: List[float] = []
    power_list: List[float] = []
    energy_list: List[float] = []

    with open(path, 'r') as fin:
        reader = csv.reader(fin, delimiter=',', quotechar='"')

        for idx, line in enumerate(reader):
            if idx == 0:
                continue  # Skip the headers

            time = float(line[0]) / 1e9  # Time in seconds
            current = float(line[1]) / 1e9  # Current in Amps
            voltage = float(line[2])   # Voltage in mV
            power = current * voltage  # Power in mW

            energy = float(line[3]) / 1e3  # Energy in mJ

            time_list.append(time)
            power_list.append(power)
            energy_list.append(energy)

    is_start = False

    start_time = None
    slope = None
    intercept = None

    start_idx: List[float] = []
    end_idx: List[float] = []

    for idx, power in enumerate(power_list):
        if (is_start) and (power > THRESHOLD):
            end_idx.append(idx)
            is_start = False
        elif (not is_start) and (power < THRESHOLD):
            start_idx.append(idx)
            is_start = True

    # Remove any trailing start times
    while len(start_idx) > len(end_idx):
        start_idx.pop(-1)

    start_idx, end_idx = filter_windows(start_idx=start_idx,
                                        end_idx=end_idx,
                                        time_list=time_list,
                                        power_list=power_list)

    start_times = [time_list[i] for i in start_idx]
    start_power_list = [power_list[i] for i in start_idx]

    end_times = [time_list[i] for i in end_idx]
    end_power_list = [power_list[i] for i in end_idx]

    fig, ax = plt.subplots()
    ax.plot(time_list, power_list)

    ax.scatter(start_times, start_power_list, color='r')
    ax.scatter(end_times, end_power_list, color='k')

    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Power (mW)')
    ax.set_title('TI MSP430 Power for Early-Exit DNNs')

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, transparent=True, bbox_inches='tight')

    energy_per_period: List[float] = []
    for start, end in zip(start_idx, end_idx):
        energy_delta = energy_list[end] - energy_list[start]
        energy_per_period.append(energy_delta)

    return energy_per_period


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv-file', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    energy_per_period = get_energy_per_period(path=args.csv_file, output_file=args.output_file)
    predicted_decisions = [int(energy > 0.03) for energy in energy_per_period]

    print('Energy per period: {}'.format(energy_per_period))
    print('Exit Decisions: {}'.format(predicted_decisions))

