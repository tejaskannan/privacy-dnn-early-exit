import csv
import os.path
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from typing import List
from privddnn.utils.file_utils import read_json


THRESHOLD = 0.13


def get_energy(path: str, num_trials: int):
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

    period_energy: List[float] = []
    start_time = None
    start_energy = None

    start_times: List[int] = []
    end_times: List[int] = []

    for time, power, energy in zip(time_list, power_list, energy_list):
        if (power > THRESHOLD) and (start_time is None):
            start_time = time
            start_energy = energy
            start_times.append(time)
        elif (power < THRESHOLD) and (start_time is not None):
            start_time = None
            period_energy.append(energy - start_energy)
            end_times.append(time)

    start_times = start_times[0:len(end_times)]

    largest_idx = np.argsort(period_energy)[::-1][0:num_trials]

    start_times = [start_times[i] for i in largest_idx]
    end_times = [end_times[i] for i in largest_idx]
    period_energy = [period_energy[i] for i in largest_idx]

    fig, ax = plt.subplots()
    ax.plot(time_list, power_list)

    for t in start_times:
        ax.axvline(t, color='orange')

    for t in end_times:
        ax.axvline(t, color='red')

    plt.show()

    return period_energy


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    directory, file_name = os.path.split(args.path)
    parameters = read_json(os.path.join(directory, file_name.replace('.csv', '.json')))
    ops_per_trial = parameters['ops_per_trial']
    num_trials = parameters['num_trials']

    result = get_energy(path=args.path, num_trials=num_trials)
    energy_per_op = [energy / ops_per_trial for energy in result]
    avg_energy_per_op = sum(energy_per_op) / len(energy_per_op)

    print('Approx Energy / Op: {:.7f}mJ'.format(avg_energy_per_op))

