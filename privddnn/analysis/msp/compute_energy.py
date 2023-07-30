import csv
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Tuple

from privddnn.utils.file_utils import save_json_gz


POWER_THRESHOLD = 0.41
PEAK_THRESHOLD = 40


def get_power(path: str) -> List[Tuple[float, float]]:
    time_list: List[float] = []
    power_list: List[float] = []
    energy_list: List[float] = []

    with open(path, 'r') as fin:
        reader = csv.reader(fin, delimiter=',', quotechar='"')
        for idx, line in enumerate(reader):
            if idx > 0:
                t = float(int(line[0]) / 1e9)  # Time in seconds
                current = float(line[1]) / 1e9  # Current in amps
                voltage = float(line[2])  # Voltage in mV
                power = current * voltage  # Power in mW
                energy = float(line[3]) / 1e3  # Energy in mJ

                time_list.append(t)
                power_list.append(power)
                energy_list.append(energy)
                    
    return time_list, power_list, energy_list


def split_periods(times: List[float], power: List[float], energy: List[float]) -> Tuple[List[float], List[float]]:
    start_time = None
    start_energy = None
    has_hit_peak = False

    start_times: List[float] = []
    end_times: List[float] = []
    period_energy: List[float] = []

    for idx, (t, pwr, e) in enumerate(zip(times, power, energy)):
        if (start_time is None) and (pwr < POWER_THRESHOLD) and (t > 5):
            start_time = t
            start_energy = e
            start_times.append(start_time)
            has_hit_peak = False
        elif (start_time is not None):
            if (pwr < POWER_THRESHOLD) and (has_hit_peak):
                end_time = t
                end_times.append(end_time)
                period_energy.append(e - start_energy)
                start_time = None
                start_energy = None
                has_hit_peak = False
            elif (pwr >= PEAK_THRESHOLD):
                has_hit_peak = True

    if len(start_times) == (len(end_times) + 1):
        start_times.pop(-1)

    return start_times, end_times, period_energy


def remove_outliers(energy: List[float]) -> List[float]:
    median = np.median(energy)
    iqr = np.percentile(energy, 75) - np.percentile(energy, 25)
    cutoff = median + 1.5 * iqr

    return [e for e in energy if e <= cutoff]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--energy-trace-file', type=str, required=True)
    args = parser.parse_args()

    times, power, energy = get_power(args.energy_trace_file)
    start_times, end_times, period_energy = split_periods(times=times, power=power, energy=energy)

    period_energy = remove_outliers(period_energy)

    output_file = args.energy_trace_file.replace('.csv', '.json.gz')
    save_json_gz(period_energy, output_file)

    fig, ax = plt.subplots()
    ax.hist(period_energy)
    plt.show()

    avg = np.mean(period_energy)
    stddev = np.std(period_energy)
    median = np.median(period_energy)
    iqr = np.percentile(period_energy, 75) - np.percentile(period_energy, 25)
    print('Median Energy: {:.4f}, IQR: {:.4f}'.format(median, iqr))
    print('Avg Energy: {:.4f} ({:.4f})'.format(avg, stddev))

    fig, ax = plt.subplots()
    ax.plot(times, power)

    for t in start_times:
        ax.axvline(t, color='red')

    for t in end_times:
        ax.axvline(t, color='orange')

    plt.show()
