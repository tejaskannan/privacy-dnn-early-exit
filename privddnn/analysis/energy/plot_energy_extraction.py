import csv
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Tuple

from privddnn.utils.plotting import PLOT_STYLE, TITLE_FONT, AXIS_FONT, LEGEND_FONT, LABEL_FONT, POLICY_LABELS, COLORS
from privddnn.analysis.msp_comp_energy import extract_energy_per_sample, TIME_DELTA, Point


matplotlib.rc('pdf', fonttype=42)
plt.rcParams['pdf.fonttype'] = 42

ADJUSTMENT = 2


def get_power(path: str, start_idx: int, end_idx: int) -> Tuple[List[float], List[float], List[float]]:
    time_list: List[float] = []
    power_list: List[float] = []
    energy_list: List[float] = []

    with open(path, 'r') as fin:
        reader = csv.reader(fin, delimiter=',', quotechar='"')

        for idx, line in enumerate(reader):
            if idx < start_idx:
                continue  # Skip the first N lines

            if idx > end_idx:
                break

            time = float(line[0]) / 1e9  # Time in seconds
            current = float(line[1]) / 1e9  # Current in Amps
            voltage = float(line[2])   # Voltage in mV
            power = current * voltage  # Power in mW
            energy = float(line[3]) / 1e3

            time_list.append(time * 1000.0)
            power_list.append(power)
            energy_list.append(energy)

    return time_list, power_list, energy_list


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trace-file', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    times, power, energy = get_power(args.trace_file, start_idx=50, end_idx=2000)
    extractions = list(extract_energy_per_sample(times, power, energy, start_time=0, time_delta=TIME_DELTA))

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.plot(times, power, linewidth=3, label='Device Power', color='#2c7fb8')

        for period_extraction in extractions:
            comp_start, comp_end = period_extraction[0], period_extraction[1]
            sample_point = period_extraction[2]
            comm_point = period_extraction[3]

            #ax.scatter(comp_start.time, comp_start.power, color='red')
            #ax.scatter(comp_end.time, comp_end.power, color='red')

            comp_times = [t for t in times if (t >= comp_start.time) and (t <= comp_end.time)]
            comp_power = [p for t, p in zip(times, power) if (t >= comp_start.time) and (t <= comp_end.time)]
            ax.plot(comp_times, comp_power, linewidth=4, label='Inference Power', color='#810f7c')

            ax.annotate('Sample', xy=(sample_point.time + 12, sample_point.power),
                        xytext=(sample_point.time + 120, sample_point.power + 1),
                        arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=2),
                        horizontalalignment='right',
                        verticalalignment='top',
                        fontsize=AXIS_FONT + ADJUSTMENT)

            ax.annotate('Bluetooth', xy=(comm_point.time + 1, comm_point.power),
                        xytext=(comm_point.time + 150, comm_point.power + 2),
                        arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=2),
                        horizontalalignment='right',
                        verticalalignment='top',
                        fontsize=AXIS_FONT + ADJUSTMENT)

            ax.annotate('Inference Energy: {:.4f}mJ'.format(period_extraction[4]), xy=(comp_start.time, comp_start.power),
                        xytext=(comp_start.time + 20, comp_start.power + 5),
                        horizontalalignment='left',
                        verticalalignment='top',
                        fontsize=AXIS_FONT + ADJUSTMENT)

        ax.tick_params(axis='both', which='major', labelsize=AXIS_FONT + ADJUSTMENT)

        ax.legend(fontsize=AXIS_FONT, loc='center')

        ax.set_xlabel('Time (ms)', fontsize=AXIS_FONT + ADJUSTMENT)
        ax.set_ylabel('Power (mW)', fontsize=AXIS_FONT + ADJUSTMENT)
        ax.set_title('TI MSP430 Power During AdNN Execution', fontsize=TITLE_FONT + ADJUSTMENT)

        plt.tight_layout()

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
