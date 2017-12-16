import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import cm
from globe import *

def mesh_plot(U, g, num):

    matplotlib.rcParams.update({'font.size': 10})

    variables = [rho, prs, vx1, vx2]
    var = ['rho', 'prs', 'vx1', 'vx2']

    for i, variable in enumerate(variables):
        plt.figure(figsize=(10,10))
        a = plt.imshow(U[variable, :, :])

        ax = plt.gca();

        # Major ticks
        ax.set_xticks(np.arange(g.lower_bc_ibeg, g.imax + 1, 10));
        ax.set_yticks(np.arange(g.lower_bc_jbeg, g.jmax + 1, 10));

        # Labels for major ticks
        ax.set_xticklabels(np.arange(g.lower_bc_ibeg, g.imax + 1, 1));
        ax.set_yticklabels(np.arange(g.lower_bc_jbeg, g.jmax + 1, 1));

        # Minor ticks
        ax.set_xticks(np.arange(-.5, g.imax + 1 + 0.5, 1), minor=True);
        ax.set_yticks(np.arange(-.5, g.jmax + 1 + 0.5, 1), minor=True);

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f'plots/grid_{var[i]}_{num:04}.png')
        plt.close()


def line_plot(U, g, num):

    f, ax1 = plt.subplots()
    ax1.plot(g.x1, U[rho])
    plt.savefig(f'plots/1D/density/line_density_{num:04}.png')
    plt.close()

    f, ax1 = plt.subplots()
    ax1.plot(g.x1, U[prs])
    plt.savefig(f'plots/1D/pressure/line_pressure_{num:04}.png')
    plt.close()

    f, ax1 = plt.subplots()
    ax1.plot(g.x1, U[vx1])
    plt.savefig(f'plots/1D/velocity_x1/line_velocity_x1_{num:04}.png')
    plt.close()


