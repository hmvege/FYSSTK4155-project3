import numpy as np
import matplotlib.pyplot as plt
import os


class CorrelatorAnalysis:
    def __init__(self, folder_name):
        # Columns should consists of MC data, where each column is a datapoint
        self.data, self.N_lattice_size, self.N_correlators = \
            load_correlator(folder_name)
        self.x = np.arange(self.N_lattice_size)

    def run_bootstraps(self, N_BS):
        # Setting up arrays
        G_bs = np.zeros((self.N_lattice_size, N_BS))     # Correlator, G
        # Correlator std, G - redundant
        G_bs_std = np.zeros((self.N_lattice_size, N_BS))
        eff_mass = np.zeros((self.N_lattice_size, N_BS))  # Temp effective mass
        # Final effective masses, averaged
        self.aM = np.zeros(self.N_lattice_size)
        # Final effective masses, std
        self.aM_std = np.zeros(self.N_lattice_size)
        # Index list for bootstrap - such that each data point uses the same index list
        index_lists = np.random.randint(
            self.N_correlators, size=(N_BS, self.N_correlators))
        # Performing the bootstrap samples
        for i in range(self.N_lattice_size):
            for j in range(N_BS):
                G_bs[i, j] = np.mean(self.data[i][index_lists[j]])
        # Computing the correlator
        eff_mass = np.log(G_bs/np.roll(G_bs, -1, axis=0))
        # Performing average and stds of the correlator
        self.aM = np.mean(eff_mass, axis=1)
        self.aM_std = np.std(eff_mass, axis=1)
        # Updating class variables
        self.N_BS = N_BS

    def plot(self, filename):
        plt.errorbar(self.x-1, y=self.aM, yerr=self.aM_std,
                     fmt="o", ecolor="r", color="0")
        plt.grid(True)
        plt.xlim(0, 21)
        plt.ylim(0.5, 1)
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel(r'$aM_N = \ln\frac{G(t)}{G(t+a)}$', fontsize=18)
        plt.title(r'Effective mass $aM$, %s bootstraps' % self.N_BS)
        # plt.savefig('%s.pdf' % filename)
        print('figures/%s.pdf written' % filename)
        plt.show()

    def plot_raw(self):
        plt.figure()
        dat = np.mean(self.data, axis=1)
        dat_std = np.std(self.data, axis=1)
        G = np.zeros(self.N_correlators)
        G_std = np.zeros(self.N_correlators)
        for i in xrange(self.N_correlators):
            index = (i + 1) % self.N_correlators
            G[i] = np.log(dat[i]/dat[index])  # Mod boundary conditions
            G_std[i] = np.sqrt((dat_std[i]/dat[i])**2 +
                               (dat_std[index]/dat[index])**2)
        plt.errorbar(self.x, G, G_std, fmt="o", ecolor="r", color="0")
        plt.grid(True)
        plt.xlim(1, 21)
        plt.ylim(0.5, 1)
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel(r'$aM_N = \ln\frac{G(t)}{G(t+a)}$', fontsize=18)
        plt.title(r'Effective mass $aM$, 0 bootstraps')
        fig_name = "figures/G_non_BS.pdf"
        plt.savefig(fig_name)
        print('{} written'.format(fig_name))
        plt.show()

    def write_bs_data(self, fname):
        dat = np.zeros((len(self.x), 2))
        for i in xrange(len(self.x)):
            dat[i, 0] = self.x[i]
            dat[i, 1] = self.aM[i]
        np.savetxt(fname, dat, fmt=["%5g", "%10g"])


def load_correlator(folder_name):
    datafiles = os.listdir(folder_name)
    correlators = np.transpose(np.asarray(
        [np.loadtxt(folder_name + '/' + filename)[:, 1]
         for filename in datafiles]))
    N_lattice_size, N_correlators = correlators.shape
    return correlators, N_lattice_size, N_correlators

def fitfunc(nt, ck, Ek):
    f_sum = 0
    for k in range(len(ck)):
        f_sum += ck[k]*np.exp(-nt*Ek[k])
    return f_sum

def chi2(C, ck, Ek, nmin=1, nmax=None):
    if isinstance(nmax, type(None)):
        nmax = len(C)
    chi_sum = 0
    for nt in range(nmin, nmax):
        for ntp in range(nmin, nmax):
            chi_sum += (C[nt] - fitfunc(nt, ck, Ek)) * w_cov(C[nt],C[ntp]) * (C[ntp] - fitfunc(ntp, ck, Ek))
    return chi_sum

def w_cov(C, Cp):
    return np.linalg.inv(np.cov([C, Cp]))
    

def main():
    folder_name = ("../data/CorrelatorAndQForAhmed/cfunForAhmed/"
                   "Kud01370000Ks01364000/twoptRandT/twoptsm64si64/")

    correlators, N_lattice_size, N_correlators = load_correlator(folder_name)
    print (correlators.shape)
    
    print(w_cov(correlators[0], correlators[0]))
    print(chi2(correlators, [1.0], [0.2], 1, 10))

    analysis = CorrelatorAnalysis(folder_name)
    analysis.run_bootstraps(250)
    analysis.plot("tmp")


if __name__ == '__main__':
    main()
