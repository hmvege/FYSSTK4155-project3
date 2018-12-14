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


def load_finite_difference_data(folder):
    pass


def plotFWData(data):
    pass


def plotComparison(tf_data, fw_data):

    # Plots results
    XX, TT = np.meshgrid(x_np, t_np)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    ax.set_title("Solution from the deep neural network w/ %d layer" %
                 len(num_hidden_neurons))
    s = ax.plot_surface(XX, TT, G_dnn, linewidth=0,
                        antialiased=False, cmap=cm.viridis)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Position $x$")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    ax.set_title("Analytical solution")
    s = ax.plot_surface(XX, TT, G_analytic, linewidth=0,
                        antialiased=False, cmap=cm.viridis)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Position $x$")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    ax.set_title("Difference")
    s = ax.plot_surface(XX, TT, diff, linewidth=0,
                        antialiased=False, cmap=cm.viridis)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Position $x$")
    plt.show()


def generateTFTableData(tf_data):
    pass


def generateTFDropoutTableData(tf_data):
    pass


def main():
    # Retrieving data
    timing_file = "../results/fw_euler_timing.dat"
    FW_timing_data = load_finite_difference_timing(timing_file)

    tf_file = "../results/testrun.json"
    tf_data = load_json(fw_file)

    fw_data_file_folder = ("/Users/hansmathiasmamenvege/Programming"
                           "/COMPHYS1/projects/project5/Diffusion")
    load_finite_difference_data(fw_data_file_folder)


    plotFWData(FW_timing_data)

    plotComparison(tf_data, FW_data)

    generateTFTableData(tf_data)

    generateTFDropoutTableData(tf_data)


if __name__ == '__main__':
    main()
