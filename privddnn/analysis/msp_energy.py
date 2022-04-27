import csv
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.integrate import trapz
from typing import Tuple, List


def compute_energy(path: str) -> Tuple[float, float]:
    """
    Computes the energy from the given EnergyTrace CSV file.
    
    Args:
        path: Path to the EnergyTrace CSV file
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
            energy_list.append(energy)

            time_list.append(time)
            power_list.append(power)

    avg_power = np.average(power_list)
    energy = energy_list[-1]

    #return energy, avg_power

    #energy = trapz(y=power_list, x=time_list)

    fig, ax = plt.subplots()
    ax.plot(time_list, power_list)
    plt.show()

    return energy, avg_power


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv-file', type=str, required=True)
    args = parser.parse_args()

    energy, power = compute_energy(path=args.csv_file)

    print('Energy: {:.4f}mJ'.format(energy))
    print('Power: {:.4f}mW'.format(power))
