import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from argparse import ArgumentParser
from scipy.integrate import trapz
from typing import Tuple, List, Optional, Iterable

from privddnn.utils.constants import SMALL_NUMBER


THRESHOLD = 4.3
TIME_DELTA = 3.0


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


def extract_energy_per_sample(time_list: List[int], power_list: List[float], energy_list: List[float], start_time: float, time_delta: float) -> Iterable[Tuple[int, int, float, float, float]]:
    # Get peaks in the power graph
    peaks, peak_properties = signal.find_peaks(x=power_list, height=25, distance=100)
    peak_heights = peak_properties['peak_heights']

    # Convert the peak indices into peak times
    peak_times = [time_list[i] for i in peaks]

    # Filter out all peaks before the start times
    idx = 0
    while peak_times[idx] < start_time:
        idx += 1

    while idx < len(peak_times):
        period_start = peak_times[idx]
        period_end = period_start + time_delta

        proc_start = peak_heights[idx]
        proc_start_idx = peaks[idx]

        proc_end = peak_heights[idx + 1]
        proc_end_idx = peaks[idx + 1]

        # Walk the first peak down until below the threshold
        while proc_start > THRESHOLD:
            proc_start_idx += 1
            proc_start = power_list[proc_start_idx]

        # Walk the second peak down until below the threshold
        while proc_end > THRESHOLD:
            proc_end_idx -= 1
            proc_end = power_list[proc_end_idx]

        # Get the energy associated with this processing period
        proc_energy = energy_list[proc_end_idx] - energy_list[proc_start_idx]

        proc_start_time = time_list[proc_start_idx]
        proc_end_time = time_list[proc_end_idx]

        proc_start_power = power_list[proc_start_idx]
        proc_end_power = power_list[proc_end_idx]

        yield proc_start_time, proc_end_time, proc_start_power, proc_end_power, proc_energy

        while (idx < len(peak_times)) and (peak_times[idx] < period_end):
            idx += 1


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

    #is_start = False

    #start_time = None
    #slope = None
    #intercept = None

    #start_idx: List[float] = []
    #end_idx: List[float] = []

    #for idx, power in enumerate(power_list):
    #    if (is_start) and (power > THRESHOLD):
    #        end_idx.append(idx)
    #        is_start = False
    #    elif (not is_start) and (power < THRESHOLD):
    #        start_idx.append(idx)
    #        is_start = True

    ## Remove any trailing start times
    #while len(start_idx) > len(end_idx):
    #    start_idx.pop(-1)

    #start_idx, end_idx = filter_windows(start_idx=start_idx,
    #                                    end_idx=end_idx,
    #                                    time_list=time_list,
    #                                    power_list=power_list)

    #start_times = [time_list[i] for i in start_idx]
    #start_power_list = [power_list[i] for i in start_idx]

    #end_times = [time_list[i] for i in end_idx]
    #end_power_list = [power_list[i] for i in end_idx]

    start_times: List[float] = []
    end_times: List[float] = []
    start_power_list: List[float] = []
    end_power_list: List[float] = []
    proc_energy: List[float] = []

    for start, end, start_power, end_power, energy in extract_energy_per_sample(time_list=time_list, power_list=power_list, energy_list=energy_list, start_time=4.5, time_delta=TIME_DELTA):
        start_times.append(start)
        end_times.append(end)

        start_power_list.append(start_power)
        end_power_list.append(end_power)

        proc_energy.append(energy)

    fig, ax = plt.subplots()
    ax.plot(time_list, power_list)

    ax.scatter(start_times, start_power_list, color='r')
    ax.scatter(end_times, end_power_list, color='k')

    #peak_times = [time_list[i] for i in peaks]
    #ax.scatter(peak_times, peak_properties['peak_heights'], color='r')

    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Power (mW)')
    ax.set_title('TI MSP430 Power for Early-Exit DNNs')

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, transparent=True, bbox_inches='tight')

    return proc_energy


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv-file', type=str, required=True)
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    energy_per_period = get_energy_per_period(path=args.csv_file, output_file=args.output_file)[:args.num_samples]
    predicted_decisions = [int(energy > 0.27) for energy in energy_per_period]

    print('Energy per period: {}'.format(energy_per_period))
    print('Exit Decisions: {}'.format(predicted_decisions))

