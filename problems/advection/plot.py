import matplotlib.pyplot as plt
import numpy as np
from settings import *


def line_plot():

    for num in range(0, 11):

        f, ax1 = plt.subplots()

        V, x1 = np.load(f'output/1D/data.{num:04}.npy', allow_pickle=True)

        ax1.plot(x1, V[rho], 'b', label=r'$\rho$')
        ax1.plot(x1, V[prs], 'r', label=r'$p$')
        ax1.plot(x1, V[vx1], 'g', label=r'$v_{x1}$')

        V2 = np.zeros_like(V[rho])
        V2[x1 < 0.25] = 1.0
        V2[x1 > 0.25] = 5.0
        V2[x1 > 0.75] = 1.0

        ax1.plot(x1, V2, 'k', label=r'$\rho$ (init)')

        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$\rho, p, v_{x1}$')

        plt.legend()
        plt.savefig(f'output/plots/line_{num:04}.png')
        plt.close()

    #np.save('2048', (V2 - V[rho]))

    print(len(x1), np.absolute((V2 - V[rho])).sum()/len(x1))


if __name__ == "__main__":
    line_plot()
