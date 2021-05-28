import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from settings import *


def line_plot():

    for num in tqdm(range(0, len(os.listdir('output/1D/')))):

        f, ax1 = plt.subplots()

        #x1p, x2p, Prho, Pvx1, Pprs = np.loadtxt(
        #    f'output/pluto/data.{num:04}.tab', unpack=True)
        V, x1 = np.load(f'output/1D/data_1D_{num:04}.npy', allow_pickle=True)

        ax1.plot(x1, V[rho], 'b', label=r'$\rho$')
        ax1.plot(x1, V[prs], 'r', label=r'$p$')
        ax1.plot(x1, V[vx1], 'g', label=r'$v_{x1}$')
        # ax1.plot(x1p, Prho, 'b--')
        # ax1.plot(x1p, Pprs, 'r--')
        # ax1.plot(x1p, Pvx1, 'g--')

        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$\rho, p, v_{x1}$')

        ax1.set_ylim(-0.1, 1.1)

        plt.legend()

        if not os.path.isdir(f'output/plots'):
            os.makedirs(f'output/plots')

        plt.savefig(f'output/plots/line_{num:04}.png')
        plt.close()


if __name__ == "__main__":
    line_plot()
