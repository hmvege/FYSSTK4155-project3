import matplotlib.pyplot as plt
import numpy as np
import json

# from tqdm import tqdm, trange

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

def load_json(filename):
    json_dict = json.load(open(filename, "r"))
    return json_dict

def load_finite_difference_timing(timing_file):
    """Loads fw finite difference data."""
    # Nx Nt alpha timing_file
    return np.loadtxt(timing_file, skiprows=1)

def load_finite_difference_data(timing_file):
    pass

help(np.loadtxt)
print(load_finite_difference_timing("../results/fw_euler_timing.dat"))

help(sorted)

def plotFWData(data):
    pass


def main():
    timing_file = "../results/fw_euler_timing.dat"
    FW_timing_data = load_finite_difference_timing(timing_file)
    plotFWData(FW_timing_data)

if __name__ == '__main__':
    main()