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


def f_analytical(x, t):
    return np.sin(np.pi*x)*np.exp(-np.pi*np.pi*t)


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


def create_folder(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)
        print("> mkdir {}".format(folder))


def generate_figure_name(base, output_folder, file_extension=".pdf",
                         **kwargs):
    fname = os.path.join(output_folder, base)
    for k, v in kwargs.items():
        fname += "_"
        fname += k + str(v)
    return fname + file_extension


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
    print("Fit parameters for Forward-Euler timing: ", res)

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
    print("Fit parameters for TensorFlow timing: ", res)

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


def plotFW3DData(data, analytical_y, output_folder="../fig/fw_3d"):
    """
    Plots forward Euler in 3D and compares with the analytical results.
    """
    create_folder(output_folder)

    for i, fw_ in enumerate(data["data"]):
        Y_fw = np.array(fw_["data"])

        if Y_fw.shape[0] > 10000 or Y_fw.shape[1] > 500:
            continue

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
        forward_euler_fig_name = generate_figure_name(
            "fw_3d", output_folder, Nt=Y_fw.shape[0])
        fig.savefig(forward_euler_fig_name)
        print_savefig(forward_euler_fig_name)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")
        ax.set_title("Analytical solution")
        s = ax.plot_surface(XX, TT, Y_analytic, linewidth=0,
                            antialiased=False, cmap=cm.viridis)
        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel(r"Time $t$")
        ax.grid(True)
        ax.view_init(elev=10., azim=45)
        analytical_fig_name = generate_figure_name(
            "analytical_3d", output_folder, Nt=Y_fw.shape[0])
        fig.savefig(analytical_fig_name)
        print_savefig(analytical_fig_name)
        plt.close(fig)

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
        fw_ana_diff_fig_name = generate_figure_name(
            "fw_ana_diff_3d", output_folder, Nt=Y_fw.shape[0])
        fig.savefig(fw_ana_diff_fig_name)
        print_savefig(fw_ana_diff_fig_name)
        plt.close(fig)


def plotDNN3DData(data, analytical_y, output_folder="../fig/dnn_3d"):
    """
    Plots DNN in 3D and compares with the analytical results.
    """
    create_folder(output_folder)

    for i, dnn_ in enumerate(data["data"]):
        Y_dnn = np.array(dnn_["G_dnn"])

        Nlayers = len(dnn_["hidden_layers"])
        neurons = int(dnn_["hidden_layers"][0])
        activation = dnn_["activation"]
        optimizer = dnn_["optimizer"]
        dropout = str(dnn_["dropout"]).replace(".", "_")

        # Forward Euler timing data
        # step-size vs time
        fixed_Nx = 10
        fixed_Nt = 10
        dnn_values = []

        # continue
        x_np = np.linspace(0.0, 1.0, Y_dnn.shape[1])
        t_np = np.linspace(0.0, 0.5, Y_dnn.shape[0])
        XX, TT = np.meshgrid(x_np, t_np)

        Y_analytic = f_analytical(XX, TT)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")
        ax.set_title("DNN")
        s = ax.plot_surface(XX, TT, Y_dnn, linewidth=0,
                            antialiased=False, cmap=cm.viridis)
        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel(r"Time $t$")
        ax.grid(True)
        ax.view_init(elev=10., azim=45)
        dnn_fig_name = generate_figure_name(
            "dnn_3d",
            output_folder,
            Nt=Y_dnn.shape[0],
            Nlayers=Nlayers,
            neurons=neurons,
            act=activation,
            opt=optimizer,
            dropout=dropout)

        fig.savefig(dnn_fig_name)
        print_savefig(dnn_fig_name)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")
        ax.set_title("Analytical solution")
        s = ax.plot_surface(XX, TT, Y_analytic, linewidth=0,
                            antialiased=False, cmap=cm.viridis)
        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel(r"Time $t$")
        ax.grid(True)
        ax.view_init(elev=10., azim=45)
        analytical_fig_name = generate_figure_name(
            "analytical_3d",
            output_folder,
            Nt=Y_dnn.shape[0],
            Nlayers=Nlayers,
            neurons=neurons,
            act=activation,
            opt=optimizer,
            dropout=dropout)

        fig.savefig(analytical_fig_name)
        print_savefig(analytical_fig_name)
        plt.close(fig)

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
        dnn_ana_diff_fig_name = generate_figure_name(
            "dnn_ana_diff_3d",
            output_folder,
            Nt=Y_dnn.shape[0],
            Nlayers=Nlayers,
            neurons=neurons,
            act=activation,
            opt=optimizer,
            dropout=dropout)

        fig.savefig(dnn_ana_diff_fig_name)
        print_savefig(dnn_ana_diff_fig_name)
        plt.close(fig)


def plot_2D_DNN(tf_data, fw_data, plot_value_name,
                activation="tanh",
                hidden_layers=[100],
                time_slices=[0.0, 0.2, 0.4]):
    """
    Include 2D plots of evolution and difference.
    """
    output_folder = "../fig/2d_dnn"
    create_folder(output_folder)

    # Forward Euler results for Nt = 5000
    fw_selected = fw_data["data"][1]

    # Selects data
    for tf_ in tf_data["data"]:
        if tf_["optimizer"] == "adam" and \
                tf_["activation"] == activation and \
                tf_["dropout"] == 0.0 and \
                tf_["hidden_layers"] == hidden_layers:
            dnn_ = tf_

    # print(fw_data["data"][1].keys())
    # exit(1)

    x_dnn = np.linspace(0, 0.5, 10)
    y_dnn = np.array(dnn_["G_dnn"])

    # Selected parameters to run for
    Nlayers = len(dnn_["hidden_layers"])
    neurons = int(dnn_["hidden_layers"][0])
    activation = dnn_["activation"]
    optimizer = dnn_["optimizer"]
    dropout = str(dnn_["dropout"]).replace(".", "_")

    x_fw = np.linspace(0, 1.0, fw_selected["Nx"])
    y_fw = np.array(fw_selected["data"])
    Nt_fw = fw_selected["Nt"]

    y_ana_dnn = np.array(dnn_["G_analytic"])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for t in time_slices:
        if plot_value_name == "dnn":
            # DNN results
            ax.plot(x_dnn, y_dnn[int(t*10), :],
                    label=r"DNN $t={0:.1f}$".format(t))

        elif plot_value_name == "ana_dnn":
            # Analytical results
            ax.plot(x_dnn, y_ana_dnn[int(t*10), :],
                    ls="--", label=r"Analytical $t={0:.1f}$".format(t))

        elif plot_value_name == "ana_fw":
            # Analytical results
            ax.plot(x_fw, f_analytical(x_fw, t),
                    ls="--", label=r"Analytical $t={0:.1f}$".format(t))

        elif plot_value_name == "fw":
            # FW results
            ax.plot(x_fw, y_fw[int(t*Nt_fw), :],
                    label=r"FW $t={0:.1f}$".format(t))

        elif plot_value_name == "fw_vs_ana":
            # Difference FW-Analytical
            ax.plot(x_fw, y_fw[int(t*Nt_fw), :] - f_analytical(x_fw, t),
                    label=r"$t={0:.1f}$".format(t))

        elif plot_value_name == "dnn_vs_ana":
            # Difference DNN-Analytical
            t_slices = int(t*10)
            ax.plot(x_dnn, y_dnn[t_slices, :] - y_ana_dnn[t_slices, :],
                    label=r"$t={0:.1f}$".format(t))
        else:
            raise UserWarning("Bad usage")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(t,x)$")
    ax.legend()
    figname = generate_figure_name(
        "dnn_time_slices_2d", output_folder,
        Nlayers=Nlayers,
        neurons=neurons,
        act=activation,
        opt=optimizer,
        dropout=dropout,
        plot_type=plot_value_name)
    fig.savefig(figname)
    print_savefig(figname)
    plt.close(fig)


def generateDNNTableData(
        tf_data, optimizer="adam",
        table_filename="../results/dnn_general_table_adam.dat"):
    """Generates data for pgfplotstable."""
    table_length = 0

    layers_to_skip = [[10], [20], [50], [10, 10],
                      [20, 20], [40, 40], [10, 10, 10], [20, 20, 20]]

    with open(table_filename, "w") as f:
        # header = (r"{$Optimizer$} {$Activation$} {$Layers$} "
        #           r"{$\mathrm{max}(\varepsilon_{\mathrm{abs}}|)$} "
        #           r"{$R^2$} {$MSE$} {$\Delta t$}")
        header = "activation layers neurons diff r2 mse"
        f.write(header)
        f.write("\n")
        for i, tf_ in enumerate(sorted(tf_data["data"],
                                       key=lambda k: (k["optimizer"],
                                                      k["activation"]))):

            if tf_["dropout"] != 0.0:
                continue

            if (tf_["activation"] == "leaky_relu" or
                    tf_["activation"] == "relu") and \
                    tf_["hidden_layers"] in layers_to_skip:
                continue

            if tf_["optimizer"] != optimizer:
                continue

            table_length += 1

            print(
                # "{0:15s}".format(tf_["optimizer"].capitalize()),
                "{0:15s}".format(
                    "{"+tf_["activation"].capitalize().replace("_", " ")+"}"),
                # "{0:40s}".format("{" + ", ".join(
                #     [str(s_) for s_ in tf_["hidden_layers"]]) + "}"),
                "{0:10d}".format(len(tf_["hidden_layers"])),
                "{0:10d}".format(tf_["hidden_layers"][0]),
                "{0:10f}".format(tf_["max_diff"]),
                "{0:10f}".format(tf_["r2"]),
                "{0:10f}".format(tf_["mse"]),
                # "{0:10f}".format(tf_["duration"]),
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
        header = "optimizer activation layers neurons diff r2 mse dropout"
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
                # "{0:15s}".format("{" + ", ".join(
                #     [str(s_) for s_ in tf_["hidden_layers"]]) + "}"),
                "{0:10d}".format(len(tf_["hidden_layers"])),
                "{0:10d}".format(tf_["hidden_layers"][0]),
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
    cost_values = np.loadtxt("../results/cost_values.dat")
    error_values = np.loadtxt("../results/max_diff_values.dat")
    epochs = np.arange(len(cost_values))*100
    assert len(cost_values) == len(error_values)

    # Note! Skipped every 100 initially, with every 50th of that its
    # every 5000th skipped.
    skip = 10

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.semilogy(epochs[::skip], cost_values[::skip], label="Cost function")
    ax1.set_ylabel(r"Cost")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(212)
    ax2.semilogy(epochs[::skip], error_values[::skip], label=r"Maximum error")
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

    tf_optimal_fpath = "../results/OptimalParametersRun1_1000000iter.json"
    tf_optimal_data = load_json(tf_optimal_fpath)

    # fw_data_file_folder = ("/Users/hansmathiasmamenvege/Programming"
    #                        "/COMPHYS1/projects/project5/Diffusion/output")
    # fw_data_file_folder = ("../results/")
    fw_data_file_folder = ("../results/forward_euler_results")
    fw_data = load_finite_difference_data(fw_data_file_folder,
                                          try_get_pickle=True)

    plotTimingFW(fw_timing_data)
    plotTimingDNN(tf_timing_data)
    plotTimingComparison(tf_data, fw_timing_data)

    generateDNNTableData(
        tf_data, optimizer="adam",
        table_filename="../results/dnn_general_table_adam.dat")
    generateDNNTableData(
        tf_data, optimizer="gd",
        table_filename="../results/dnn_general_table_gd.dat")
    generateDNNDropoutTableData(tf_data)
    generateDNNTableData(
        tf_optimal_data, optimizer="adam",
        table_filename="../results/dnn_optimal_table_adam.dat")

    plot_error_and_cost()
    plotFW3DData(fw_data, tf_optimal_data["data"][0]["G_analytic"])
    plotDNN3DData(tf_data, tf_optimal_data["data"][0]["G_analytic"],
                  output_folder="../fig/dnn_optimal_3d")

    plot_2D_DNN(tf_optimal_data, fw_data, "fw")
    plot_2D_DNN(tf_optimal_data, fw_data, "ana_fw")
    plot_2D_DNN(tf_optimal_data, fw_data, "fw_vs_ana")
    plot_2D_DNN(tf_optimal_data, fw_data, "ana_dnn")
    plot_2D_DNN(tf_optimal_data, fw_data, "dnn")
    plot_2D_DNN(tf_optimal_data, fw_data, "dnn_vs_ana")

    exit("COMPLETED")


if __name__ == '__main__':
    main()
