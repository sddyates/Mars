import sys
sys.path.insert(0, '../../src')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from globe import *

def line_plot():

    for num in range(0, 11):

        f, ax1 = plt.subplots()

        x1p, x2p, Prho, Pvx1, Pprs = np.loadtxt(f'output/pluto/data.{num:04}.tab', unpack=True)
        V, x1 = np.load(f'output/1D/data.{num:04}.npy')

        ax1.plot(x1, V[rho], 'b')
        ax1.plot(x1, V[prs], 'r')
        ax1.plot(x1, V[vx1], 'g')
        ax1.plot(x1p, Prho, 'b--')
        ax1.plot(x1p, Pprs, 'r--')
        ax1.plot(x1p, Pvx1, 'g--')

        plt.savefig(f'output/plots/line_{num:04}.png')
        plt.close()

def mesh_plot():

    matplotlib.rcParams.update({'font.size': 10})

    for num in range(0, 200):

        V, x1, x2 = np.load(f'output/2D/data.{num:04}.npy')


        variables = [rho, prs, vx1, vx2]
        var = ['density', 'pressure', 'velocity_x1', 'velocity_x2']

        for i, variable in enumerate(variables):
            plt.figure(figsize=(10,10))
            a = plt.imshow(V[variable, :, :], extent=(x1.min(), x1.max(), 
                           x2.min(), x2.max()))

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
            plt.savefig(f'output/plots/2D/{var[i]}/grid_{var[i]}_{num:04}.png')
            plt.close()


mesh_plot()
#line_plot()
