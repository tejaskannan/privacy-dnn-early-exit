import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from collections import namedtuple
from argparse import ArgumentParser
from sklearn.cluster import KMeans
from scipy.integrate import trapz
from typing import Tuple, List, Optional, Iterable

from privddnn.utils.constants import SMALL_NUMBER


THRESHOLD = 4.1
TIME_DELTA = 3.5
STABLE_THRESHOLD = 0.15
STABLE_DIST = 1000
PEAK_HEIGHT = 22
START_TIME = 5

Point = namedtuple('Point', ['time', 'power'])


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


def is_power_stable_in_range(power_list: List[float], start_idx: int, end_idx: int) -> bool:
    if start_idx >= len(power_list):
        return True

    max_value = max(power_list[start_idx:end_idx])
    return max_value <= STABLE_THRESHOLD


def extract_energy_per_sample(time_list: List[float], power_list: List[float], energy_list: List[float], start_time: float, time_delta: float) -> Iterable[Tuple[Point, Point, Point, Point, float]]:
    # Get peaks in the power graph
    raw_peaks, peak_properties = signal.find_peaks(x=power_list, height=PEAK_HEIGHT, distance=100)
    raw_peak_heights = peak_properties['peak_heights']

    # Convert the peak indices into peak times
    raw_peak_times = [time_list[i] for i in raw_peaks]

    # Filter out all peaks before the start times
    peak_idx = 0
    while raw_peak_times[peak_idx] < start_time:
        peak_idx += 1

    # Filter out all peaks that occur in the middle of the Bluetooth procedure. This assumes that the first peak is `perfect`
    peak_times: List[int] = []
    peak_heights: List[float] = []
    peaks: List[int] = []
    
    while peak_idx < (len(raw_peak_times) - 1):
        peak_times.append(raw_peak_times[peak_idx])
        peak_heights.append(raw_peak_heights[peak_idx])
        peaks.append(raw_peaks[peak_idx])

        peak_idx += 1
        peak_times.append(raw_peak_times[peak_idx])
        peak_heights.append(raw_peak_heights[peak_idx])
        peaks.append(raw_peaks[peak_idx])

        # Walk forward until we find a time in which the power is stable below 0.13
        start_idx = raw_peaks[peak_idx]
        end_idx = start_idx + STABLE_DIST

        while not is_power_stable_in_range(power_list, start_idx, end_idx):
            start_idx = end_idx
            end_idx = start_idx + STABLE_DIST

        start_time = time_list[start_idx] if start_idx < len(time_list) else time_list[-1] + 1.0

        while (peak_idx < len(raw_peak_times)) and (raw_peak_times[peak_idx] < start_time):
            peak_idx += 1

    idx = 0
    while idx < (len(peak_times) - 1):
        #period_start = peak_times[idx]
        #period_end = period_start + time_delta

        proc_start_power = peak_heights[idx]
        proc_start_idx = peaks[idx]

        proc_end_power = peak_heights[idx + 1]
        proc_end_idx = peaks[idx + 1]

        # Walk the first peak down until below the threshold
        while proc_start_power > THRESHOLD:
            proc_start_idx += 1
            proc_start_power = power_list[proc_start_idx]

        # Walk the second peak down until below the threshold
        while proc_end_power > THRESHOLD:
            proc_end_idx -= 1
            proc_end_power = power_list[proc_end_idx]

        # Get the energy associated with this processing period
        proc_energy = energy_list[proc_end_idx] - energy_list[proc_start_idx]

        proc_start = Point(time=time_list[proc_start_idx], power=power_list[proc_start_idx])
        proc_end = Point(time=time_list[proc_end_idx], power=power_list[proc_end_idx])

        peak_start = Point(time=peak_times[idx], power=peak_heights[idx])
        peak_end = Point(time=peak_times[idx + 1], power=peak_heights[idx + 1])

        yield proc_start, proc_end, peak_start, peak_end, proc_energy

        idx += 2


def get_energy_per_period(path: str, output_file: Optional[str], should_plot: bool) -> List[float]:
    """
    Computes the energy from the given EnergyTrace CSV file.
    
    Args:
        path: Path to the EnergyTrace CSV file
        output_file: Optional path to output file for the plotted result
        should_plot: Whether to plot the result
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

    start_points: List[Point] = []
    end_points: List[Point] = []
    start_peaks: List[Point] = []
    end_peaks: List[Point] = []
    proc_energy: List[float] = []

    energy_iterator = extract_energy_per_sample(time_list=time_list, power_list=power_list, energy_list=energy_list, start_time=START_TIME, time_delta=TIME_DELTA)

    for proc_start, proc_end, peak_start, peak_end, energy in energy_iterator:
        start_points.append(proc_start)
        end_points.append(proc_end)

        start_peaks.append(peak_start)
        end_peaks.append(peak_end)

        proc_energy.append(energy)

    if should_plot:
        fig, ax = plt.subplots()

        subsampled_time = [t for i, t in enumerate(time_list) if i % 10 == 0]
        subsampled_power = [p for i, p in enumerate(power_list) if i % 10 == 0]

        ax.plot(subsampled_time, subsampled_power)

        ax.scatter([p.time for p in start_points], [p.power for p in start_points], color='r')
        ax.scatter([p.time for p in end_points], [p.power for p in end_points], color='k')

        #for start_peak in start_peaks:
        #    ax.annotate('Sample', xy=start_peak, xytext=(start_peak.time - 0.3, start_peak.power + 1.0))

        #for end_peak in end_peaks:
        #    ax.annotate('Bluetooth On', xy=end_peak, xytext=(end_peak.time - 0.3, end_peak.power + 1.0))

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


def get_exit_points(energy: List[float], num_outputs: int) -> List[int]:
    model = KMeans(n_clusters=num_outputs)
    preds = model.fit_predict(X=np.reshape(energy, (-1, 1)), y=None)

    centers = model.cluster_centers_.reshape(-1)  # [D]
    center_order = np.argsort(centers).astype(int).tolist()  # [D]
    center_labels = [center_order.index(i) for i in range(len(centers))]  # [D]

    return [center_labels[p] for p in preds]   


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv-file', type=str, required=True)
    parser.add_argument('--num-samples', type=int, required=True)
    parser.add_argument('--num-outputs', type=int, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    energy_per_period = get_energy_per_period(path=args.csv_file, output_file=args.output_file, should_plot=True)[:args.num_samples]
    predicted_decisions = get_exit_points(energy=energy_per_period, num_outputs=args.num_outputs)

    print('Energy per period: {}'.format(energy_per_period))
    print('Exit Decisions: {}'.format(predicted_decisions))

