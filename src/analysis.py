import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
import pickle

# from tqdm import tqdm, trange

from matplotlib import cm, rc, rcParams
from mpl_toolkits.mplot3d import axes3d

# rc("text", usetex=True)
# rc("font", **{"family": "sans-serif", "serif": ["Computer Modern"]})
# rcParams["font.family"] += ["serif"]


def load_pickle(pickle_file_name):
    """Loads a pickle from given pickle_file_name."""
    with open(pickle_file_name, "rb") as f:
        data = pickle.load(f)
        print("Pickle file loaded: {}".format(pickle_file_name))
    return data


def save_pickle(pickle_file_name, data):
    """Saves data as a pickle."""
    with open(pickle_file_name, "wb") as f:
        pickle.dump(data, f)
        print("Data pickled and dumped to: {:s}".format(pickle_file_name))


def load_json(filename):
    json_dict = json.load(open(filename, "r"))
    print("Json data {} loaded.".format(filename))
    return json_dict


def load_finite_difference_timing(timing_file):
    """Loads fw finite difference data."""
    # Nx Nt alpha timing_file
    print("Finite difference timing data {} loaded.".format(timing_file))
    return np.loadtxt(timing_file, skiprows=1)


def load_finite_difference_data(folder, try_get_pickle=True):
    """Loads Forward Euler finite difference data."""

    # Checks if pickle exists, and if so loads that instead
    pickle_fname = "fw_data_picle.pkl"
    if os.path.isfile(pickle_fname) and try_get_pickle:
        data_dict = load_pickle(pickle_fname)
        print("Forward Euler data retrieved.")
        return data_dict

    data_dict = {
        "data": []
    }

    for filename in sorted(os.listdir(folder))[::-1]:
        if filename.startswith("."):
            continue

        fname_lst = filename.split(".dat")[0].split("_")

        # Some of the files are from other runs
        if len(fname_lst) <= 4:
            continue

        Nx = int(fname_lst[2].lstrip("Nx"))
        Nt = int(fname_lst[3].lstrip("Nt"))
        alpha = float(fname_lst[4].lstrip("alpha"))

        print("Loading {0:s}".format(filename))
        data = np.loadtxt(os.path.join(folder, filename))

        _tmp = {
            "data": data,
            "Nx": Nx,
            "Nt": Nt,
            "alpha": alpha,
        }

        data_dict["data"].append(_tmp)

    # Saves pickle
    save_pickle(pickle_fname, data_dict)

    print("Forward Euler data retrieved.")
    return data_dict


def plotTimingFW(fw_timing_data, figure_name="../fig/timing_fw.pdf"):
    """
    Forward Euler timing data.
    step-size vs time.
    """
    fixed_Nx = 10
    fixed_Nt = 10
    fw_values = []
    for fw_ in fw_timing_data:

        # Selects only thos with a certain Nx
        if np.abs(fw_[0] - fixed_Nx) < 1e-16:

            # Selects only those which has a certain Ny
            if np.abs((fw_[1] * fw_[2]) - fixed_Nt) < 1e-16:
                fw_values.append(fw_)

    fw_values = np.array(fw_values)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.loglog(fw_values[:, 2], fw_values[:, 2],
               label=r"Forward-Euler")
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"$t$[s]")
    ax1.grid(True)
    ax1.legend()
    fig1.savefig(figure_name)
    plt.close(fig1)
    print("Plotted {}".format(figure_name))


def plotTimingComparison(tf_data, fw_timing_data):
    """Compares tensor flow timing data to Forward Euler data."""
    # Forward Euler timing data
    # step-size vs time
    fixed_Nx = 10
    fixed_Nt = 10
    fw_values = []
    for fw_ in fw_timing_data:

        # Selects only thos with a certain Nx
        if np.abs(fw_[0] - fixed_Nx) < 1e-16:

            # Selects only those which has a certain Ny
            if np.abs((fw_[1] * fw_[2]) - fixed_Nt) < 1e-16:
                fw_values.append(fw_)

    fw_values = np.array(fw_values)

    tf_values = []
    for tf_ in tf_data["data"]:
        print(tf_.keys())
        exit(1)

    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # ax1.plot()


def plotFWData(data, analytical_y):
    print(len(data["data"]))

    # Forward Euler timing data
    # step-size vs time
    fixed_Nx = 10
    fixed_Nt = 10
    fw_values = []

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)

    print (np.array(analytical_y)[0], data["data"][-1]["data"][-1])

    for fw_ in data["data"]:
        print (fw_["Nx"], fixed_Nx, fw_["Nt"], fixed_Nt)
        if (fw_["Nx"] == fixed_Nx):

            print(fw_["data"].shape)
            fw_values.append(fw_["data"])
            ax1.plot(fw_["data"][-1], label=r"$\alpha={0:.1e}$".format(fw_["alpha"]))

            ax2.semilogy(np.abs(fw_["data"][-1] - analytical_y), label=r"$\alpha={0:.1e}$".format(fw_["alpha"]))

    ax1.set_ylabel(r"$u_{\mathrm{FW}}$")
    ax2.set_ylabel(r"$|u_{\mathrm{FW}} - u_{\mathrm{Analytical}}|$")

    ax2.set_xlabel(r"$t$")

    ax1.grid(True)
    ax2.grid(True)

    # ax1.legend()
    # ax2.legend()

    plt.show()

def plotComparison(tf_data, fw_data):

    print(tf_data["data"][0].keys())
    exit(1)
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
    ax.set_grid(True)
    plt.show()


def generateTFTableData(tf_data):
    pass


def generateTFDropoutTableData(tf_data):
    pass


def main():
    # Retrieving data
    timing_file = "../results/fw_euler_timing.dat"
    # timing_file = ("/Users/hansmathiasmamenvege/Programming"
    #                "/COMPHYS1/projects/project5/Diffusion/output"
    #                "/fw_euler_timing.dat")
    fw_timing_data = load_finite_difference_timing(timing_file)

    tf_file = "../results/testrun.json"
    tf_data = load_json(tf_file)

    # fw_data_file_folder = ("/Users/hansmathiasmamenvege/Programming"
    #                        "/COMPHYS1/projects/project5/Diffusion/output")
    fw_data_file_folder = ("../results/")
    fw_data = load_finite_difference_data(fw_data_file_folder)

    plotTimingFW(fw_timing_data)
    # plotTimingComparison(tf_data, fw_timing_data)

    # plotFWData(fw_data, tf_data["data"][0]["G_analytic"])
    plotComparison(tf_data, fw_data)
    exit("COMPLETED")

    generateTFTableData(tf_data)

    generateTFDropoutTableData(tf_data)


if __name__ == '__main__':
    main()
