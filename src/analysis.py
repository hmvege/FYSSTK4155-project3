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

def print_savefig(figname):
    print("Saved figure: {}".format(figname))


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

    # print(fw_values[:, 2])
    # print(fw_values[:, 3])

    res = np.polyfit(fw_values[1:, 1][::-1], fw_values[1:, 3][::-1], 1)
    print ("Fit parameters for Forward-Euler timing: ", res)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.loglog(fw_values[:, 1][::-1], fw_values[:, 3][::-1],
               "o-", label=r"Forward-Euler")
    ax1.set_xlabel(r"$N_t$")
    ax1.set_ylabel(r"$t$[s]")
    ax1.grid(True)
    ax1.legend()


    fig1.savefig(figure_name)
    plt.close(fig1)
    print("Plotted {}".format(figure_name))


def plotTimingDNN(tf_data, figure_name="../fig/timing_tf.pdf"):
    """
    Forward Euler timing data.
    step-size vs time.
    """

    fixed_Nx = 10
    fixed_Nt = 10
    tf_layers = np.empty(len(tf_data["data"]), dtype=int)
    tf_times = np.empty(len(tf_data["data"]), dtype=float)
    for i, tf_ in enumerate(tf_data["data"]):
        tf_times[i] = tf_["duration"]
        tf_layers[i] = tf_["hidden_layers"][0]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(tf_layers, tf_times,
               "o-", label=r"DNN")
    ax1.set_xlabel(r"Layer size")
    ax1.set_ylabel(r"$t$[s]")
    ax1.grid(True)
    ax1.legend()

    res = np.polyfit(tf_layers, tf_times, 1)
    print ("Fit parameters for TensorFlow timing: ", res)

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


def plotFW3DData(data, analytical_y):
    """
    Plots forward Euler in 3D and compares with the analytical results.
    """
    for i, fw_ in enumerate(data["data"]):
        Y_fw = np.array(fw_["data"])

        # Forward Euler timing data
        # step-size vs time
        fixed_Nx = 10
        fixed_Nt = 10
        fw_values = []

        # continue
        x_np = np.linspace(0.0, 1.0, Y_fw.shape[1])
        t_np = np.linspace(0.0, 0.5, Y_fw.shape[0])
        XX, TT = np.meshgrid(x_np, t_np)

        Y_analytic = np.sin(np.pi*XX)*np.exp(-np.pi*np.pi*TT)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")
        ax.set_title("Forward-Euler")
        s = ax.plot_surface(XX, TT, Y_fw, linewidth=0,
                            antialiased=False, cmap=cm.viridis)
        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel(r"Time $t$")
        ax.grid(True)
        ax.view_init(elev=10., azim=45)
        forward_euler_fig_name = \
            "../fig/fw_3d_Nt{}.pdf".format(Y_fw.shape[0])
        fig.savefig(forward_euler_fig_name)
        print_savefig(forward_euler_fig_name)
        # plt.show()
        # exit("@230")

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")
        ax.set_title("Analytical solution")
        s = ax.plot_surface(XX, TT, Y_analytic, linewidth=0,
                            antialiased=False, cmap=cm.viridis)
        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel(r"Time $t$")
        ax.grid(True)
        ax.view_init(elev=10., azim=45)
        analytical_fig_name = \
            "../fig/fw_analytical_3d_Nt{}.pdf".format(Y_fw.shape[0])
        fig.savefig(analytical_fig_name)
        print_savefig(analytical_fig_name)

        diff = np.abs(Y_analytic - Y_fw)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")
        ax.set_title("Difference")
        s = ax.plot_surface(XX, TT, diff, linewidth=0,
                            antialiased=False, cmap=cm.viridis)
        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel(r"Time $t$")
        ax.grid(True)
        ax.view_init(elev=10., azim=45)
        fw_ana_diff_fig_name = \
            "../fig/fw_ana_diff_3d_Nt{}.pdf".format(Y_fw.shape[0])
        fig.savefig(fw_ana_diff_fig_name)
        print_savefig(fw_ana_diff_fig_name)
        plt.close(fig)

def plotDNN3DData(data, analytical_y):
    """
    Plots DNN in 3D and compares with the analytical results.
    """
    for i, dnn_ in enumerate(data["data"]):
        Y_dnn = np.array(dnn_)

        # Forward Euler timing data
        # step-size vs time
        fixed_Nx = 10
        fixed_Nt = 10
        dnn_values = []

        # continue
        x_np = np.linspace(0.0, 1.0, Y_dnn.shape[1])
        t_np = np.linspace(0.0, 0.5, Y_dnn.shape[0])
        XX, TT = np.meshgrid(x_np, t_np)

        Y_analytic = np.sin(np.pi*XX)*np.exp(-np.pi*np.pi*TT)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")
        ax.set_title("DNN")
        s = ax.plot_surface(XX, TT, Y_dnn, linewidth=0,
                            antialiased=False, cmap=cm.viridis)
        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel(r"Time $t$")
        ax.grid(True)
        ax.view_init(elev=10., azim=45)
        forward_euler_fig_name = \
            "../fig/dnn_3d_Nt{}.pdf".format(Y_dnn.shape[0])
        fig.savefig(forward_euler_fig_name)
        print_savefig(forward_euler_fig_name)
        # plt.show()
        # exit("@230")

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")
        ax.set_title("Analytical solution")
        s = ax.plot_surface(XX, TT, Y_analytic, linewidth=0,
                            antialiased=False, cmap=cm.viridis)
        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel(r"Time $t$")
        ax.grid(True)
        ax.view_init(elev=10., azim=45)
        analytical_fig_name = \
            "../fig/dnn_analytical_3d_Nt{}.pdf".format(Y_dnn.shape[0])
        fig.savefig(analytical_fig_name)
        print_savefig(analytical_fig_name)

        diff = np.abs(Y_analytic - Y_dnn)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")
        ax.set_title("Difference")
        s = ax.plot_surface(XX, TT, diff, linewidth=0,
                            antialiased=False, cmap=cm.viridis)
        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel(r"Time $t$")
        ax.grid(True)
        ax.view_init(elev=10., azim=45)
        dnn_ana_diff_fig_name = \
            "../fig/dnn_ana_diff_3d_Nt{}.pdf".format(Y_dnn.shape[0])
        fig.savefig(dnn_ana_diff_fig_name)
        print_savefig(dnn_ana_diff_fig_name)
        plt.close(fig)



def plotFWData(data, analytical_y):
    # Forward Euler timing data
    # step-size vs time
    fixed_Nx = 10
    fixed_Nt = 10
    fw_values = []

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)

    # print (np.array(analytical_y).shape)
    Y_fw = np.array(data["data"][-4]["data"])

    print(Y_fw.shape)

    x_np = np.linspace(0.0, 1.0, fixed_Nx)
    t_np = np.linspace(0.0, 0.5, Y_fw.shape[0])
    XX, TT = np.meshgrid(x_np, t_np)

    Y_analytic = np.sin(np.pi*XX)*np.exp(-np.pi*np.pi*TT)

    plt.show()
    plt.close(fig)

    exit(1)

    for fw_ in data["data"]:
        print(fw_["Nx"], fixed_Nx, fw_["Nt"], fixed_Nt)
        if (fw_["Nx"] == fixed_Nx):

            print(fw_["data"].shape)
            fw_values.append(fw_["data"])
            ax1.plot(fw_["data"][-1],
                     label=r"$\alpha={0:.1e}$".format(fw_["alpha"]))

            ax2.semilogy(np.abs(fw_["data"][-1] - analytical_y),
                         label=r"$\alpha={0:.1e}$".format(fw_["alpha"]))

    ax1.set_ylabel(r"$u_{\mathrm{FW}}$")
    ax2.set_ylabel(r"$|u_{\mathrm{FW}} - u_{\mathrm{Analytical}}|$")

    ax2.set_xlabel(r"$t$")

    ax1.grid(True)
    ax2.grid(True)

    # ax1.legend()
    # ax2.legend()

    plt.show()


def plotComparison(tf_data, fw_data):
    """
    Include 2D plots of evolution and difference.
    """

    print(tf_data["data"][0].keys())
    print(tf_data["data"][0]["G_analytic"][0])
    for i in fw_data["data"]:
        print(i["data"][-1])
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
    ax.grid(True)
    plt.show()


def generateDNNTableData(tf_data,
                        table_filename="../results/dnn_general_table.dat"):
    """Generates data for pgfplotstable."""
    table_length = 0

    with open(table_filename, "w") as f:
        # header = (r"{$Optimizer$} {$Activation$} {$Layers$} "
        #           r"{$\mathrm{max}(\varepsilon_{\mathrm{abs}}|)$} "
        #           r"{$R^2$} {$MSE$} {$\Delta t$}")
        header = "optimizer activation layers diff r2 mse time"
        f.write(header)
        f.write("\n")
        for i, tf_ in enumerate(sorted(tf_data["data"],
                                       key=lambda k: (k["optimizer"],
                                                      k["activation"]))):

            if tf_["dropout"] != 0.0:
                continue

            table_length += 1

            print(
                "{0:15s}".format(tf_["optimizer"].capitalize()),
                "{0:15s}".format(
                    "{"+tf_["activation"].capitalize().replace("_", " ")+"}"),
                "{0:40s}".format("{" + ", ".join(
                    [str(s_) for s_ in tf_["hidden_layers"]]) + "}"),
                "{0:10f}".format(tf_["max_diff"]),
                "{0:10f}".format(tf_["r2"]),
                "{0:10f}".format(tf_["mse"]),
                "{0:10f}".format(tf_["duration"]),
                file=f)

    print("Table of length {} written to file {}.".format(
        table_length, table_filename))


def generateDNNDropoutTableData(
        tf_data,
        table_filename="../results/dnn_dropout_table.dat"):
    """Generates dropout data for pgfplotstable."""
    table_length = 0

    with open(table_filename, "w") as f:
        # header = (r"{$Optimizer$} {$Activation$} {$Layers$} "
        #           r"{$\mathrm{max}(\varepsilon_{\mathrm{abs}}|)$} "
        #           r"{$R^2$} {$MSE$} {$\Delta t$}")
        header = "optimizer activation layers diff r2 mse dropout"
        f.write(header)
        f.write("\n")
        for i, tf_ in enumerate(sorted(tf_data["data"],
                                       key=lambda k: (k["optimizer"],
                                                      k["activation"]))):

            if tf_["dropout"] == 0.0:
                continue

            table_length += 1

            print(
                "{0:15s}".format(tf_["optimizer"].capitalize()),
                "{0:15s}".format(
                    "{"+tf_["activation"].capitalize().replace("_", " ")+"}"),
                "{0:15s}".format("{" + ", ".join(
                    [str(s_) for s_ in tf_["hidden_layers"]]) + "}"),
                "{0:10f}".format(tf_["max_diff"]),
                "{0:10f}".format(tf_["r2"]),
                "{0:10f}".format(tf_["mse"]),
                # "{0:10f}".format(tf_["duration"]),
                "{0:10f}".format(tf_["dropout"]),
                file=f)

    print("Table of length {} written to file {}.".format(
        table_length, table_filename))

def plot_error_and_cost():
    """Plots error and cost of neural network of the epochs."""
    cost_values = np.loadtxt("cost_values.dat")
    error_values = np.loadtxt("max_diff_values.dat")
    epochs = np.arange(len(cost_values))
    assert len(cost_values)==len(error_values)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.semilogy(epochs, cost_values, label="Cost function")
    ax1.set_ylabel("Cost")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(212)
    ax2.semilogy(epochs, error_values, label=r"Maximum error")
    ax2.set_ylabel(r"$\max(\varepsilon_{\mathrm{diff}})$")
    ax2.set_xlabel(r"Epochs")
    ax2.grid(True)
    ax2.legend()

    figname = "../fig/cost_error.pdf"
    fig.savefig(figname)
    print_savefig(figname)



def main():
    # Retrieving data
    timing_file = "../results/fw_euler_timing.dat"
    # timing_file = ("/Users/hansmathiasmamenvege/Programming"
    #                "/COMPHYS1/projects/project5/Diffusion/output"
    #                "/fw_euler_timing.dat")
    fw_timing_data = load_finite_difference_timing(timing_file)

    tf_file = "../results/testrun.json"
    tf_file = "../results/productionRun3_100000iter.json"
    tf_data = load_json(tf_file)
    tf_timing_data = load_json("../results/TimingRun1_100000iter.json")

    # fw_data_file_folder = ("/Users/hansmathiasmamenvege/Programming"
    #                        "/COMPHYS1/projects/project5/Diffusion/output")
    # fw_data_file_folder = ("../results/")
    fw_data_file_folder = ("../results/forward_euler_results")
    fw_data = load_finite_difference_data(fw_data_file_folder,
                                          try_get_pickle=True)

    # plotTimingFW(fw_timing_data)
    # plotTimingDNN(tf_timing_data)
    # plotTimingComparison(tf_data, fw_timing_data)

    plot_error_and_cost()

    exit("COMPLETED")
    plotFW3DData(fw_data, tf_data["data"][0]["G_analytic"])
    plotDNN3DData(tf_data, tf_data["data"][0]["G_analytic"])
    # plotFWData(fw_data, tf_data["data"][0]["G_analytic"])
    plotComparison(tf_data, fw_data)

    generateDNNTableData(tf_data)
    generateDNNDropoutTableData(tf_data)


if __name__ == '__main__':
    main()
