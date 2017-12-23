import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import cm
from globe import *

def numpy_dump(V, g, num):

    np.save(f'output/1D/data.{num:04}.npy', (V, g.x1))

def mesh_plot(V, g, num):

    matplotlib.rcParams.update({'font.size': 10})

    variables = [rho, prs, vx1, vx2]
    var = ['density', 'pressure', 'velocity_x1', 'velocity_x2']

    for i, variable in enumerate(variables):
        plt.figure(figsize=(10,10))
        a = plt.imshow(V[variable, :, :])

        ax = plt.gca();

        # Major ticks
        #ax.set_xticks(np.arange(g.lower_bc_ibeg, g.imax + 1, 10));
        #ax.set_yticks(np.arange(g.lower_bc_jbeg, g.jmax + 1, 10));

        # Labels for major ticks
        #ax.set_xticklabels(np.arange(g.lower_bc_ibeg, g.imax + 1, 1));
        #ax.set_yticklabels(np.arange(g.lower_bc_jbeg, g.jmax + 1, 1));

        # Minor ticks
        #ax.set_xticks(np.arange(-.5, g.imax + 1 + 0.5, 1), minor=True);
        #ax.set_yticks(np.arange(-.5, g.jmax + 1 + 0.5, 1), minor=True);

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f'output/2D/{var[i]}/grid_{var[i]}_{num:04}.png')
        plt.close()


def line_plot(V, g, num):

    f, ax1 = plt.subplots()
    ax1.plot(g.x1, V[rho])
    #plt.savefig(f'plots/1D/density/line_density_{num:04}.png')
    #plt.close()

    #f, ax1 = plt.subplots()
    ax1.plot(g.x1, V[prs])
    #plt.savefig(f'plots/1D/pressure/line_pressure_{num:04}.png')
    #plt.close()

    #f, ax1 = plt.subplots()
    ax1.plot(g.x1, V[vx1])
    #plt.savefig(f'plots/1D/velocity_x1/line_velocity_x1_{num:04}.png')
    #plt.close()

    plt.savefig(f'plots/1D/profile/line_{num:04}.png')
    plt.close()

