import numpy as np
import matplotlib.pyplot as plt
import os


def load_correlator(folder_name):
    datafiles = os.listdir(folder_name)
    correlators = np.transpose(np.asarray(
        [np.loadtxt(folder_name + '/' + filename)[:, 1]
         for filename in datafiles]))
    N_lattice_size, N_correlators = correlators.shape
    return correlators, N_lattice_size, N_correlators


def main():
    folder_name = ("../data/CorrelatorAndQForAhmed/cfunForAhmed/"
                   "Kud01370000Ks01364000/twoptRandT/twoptsm64si64/")

    correlators, N_lattice_size, N_correlators = load_data(folder_name)


if __name__ == '__main__':
    main()
