import csv
import os.path
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from scipy.integrate import trapz
from typing import List
from privddnn.utils.file_utils import read_json


THRESHOLD = 0.13


def get_energy(path: str, num_trials: int, should_plot: bool):
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

    window_power: List[float] = []
    window_times: List[int] = []
    baseline_power = np.min([p for p in power_list if p > 0])

    for time, power, energy in zip(time_list, power_list, energy_list):
        if start_time is not None:
            window_power.append(power)
            window_times.append(time)

        if (power > THRESHOLD) and (start_time is None):
            start_time = time
            start_energy = energy
            start_times.append(time)
        elif (power < THRESHOLD) and (start_time is not None):
            step_energy = energy - start_energy

            start_time = None
            start_energy = None

            #energy = trapz(y=window_power, x=window_times)
            #baseline_energy = trapz(y=[baseline_power for _ in window_power],
            #                        x=window_times)

            #energy = energy - baseline_energy

            window_power = []
            window_times = []

            period_energy.append(step_energy)
            end_times.append(time)

    start_times = start_times[0:len(end_times)]
    largest_idx = np.argsort(period_energy)[::-1][0:num_trials]

    start_times = [start_times[i] for i in largest_idx]
    end_times = [end_times[i] for i in largest_idx]
    period_energy = [period_energy[i] for i in largest_idx]

    if should_plot:
        fig, ax = plt.subplots()
        ax.plot(time_list, power_list)

        for t in start_times:
            ax.axvline(t, color='orange')

        for t in end_times:
            ax.axvline(t, color='red')

        ax.axhline(baseline_power, color='black')
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

    result = get_energy(path=args.path, num_trials=num_trials, should_plot=True)
    energy_per_op = [energy / ops_per_trial for energy in result]
    avg_energy_per_op = np.average(energy_per_op)
    std_energy_per_op = np.std(energy_per_op)

    print('Approx Energy / Op: {:.7f}mJ ({:7f})'.format(avg_energy_per_op, std_energy_per_op))

