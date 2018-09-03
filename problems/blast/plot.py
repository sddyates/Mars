import sys
sys.path.insert(0, '../../src')
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from settings import *

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

    for num in range(0, 300):

        V, x1, x2 = np.load(f'output/2D/data.{num:04}.npy')
        print(f'data.{num:04}.npy')

        variables = [rho, prs, vx1, vx2]
        var = ['density', 'pressure', 'velocity_x1', 'velocity_x2']

        for i, variable in enumerate(variables):
            plt.figure(figsize=(10,10))
            a = plt.imshow(V[variable, :, :], extent=(x1.min(), x1.max(), 
                           x2.min(), x2.max()))

            ax = plt.gca()

            # Major ticks
            #ax.set_xticks(np.arange(g.lower_bc_ibeg, g.imax + 1, 10))
            #ax.set_yticks(np.arange(g.lower_bc_jbeg, g.jmax + 1, 10))

            # Labels for major ticks
            #ax.set_xticklabels(np.arange(g.lower_bc_ibeg, g.imax + 1, 1))
            #ax.set_yticklabels(np.arange(g.lower_bc_jbeg, g.jmax + 1, 1))

            # Minor ticks
            #ax.set_xticks(np.arange(-0.5, g.imax + 1 + 0.5, 1), minor=True)
            #ax.set_yticks(np.arange(-0.5, g.jmax + 1 + 0.5, 1), minor=True)

            # Gridlines based on minor ticks
            ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
            plt.colorbar()

            plt.tight_layout()
            plt.savefig(f'output/plots/2D/{var[i]}/grid_{var[i]}_{num:04}.png')
            plt.close()


def mesh_plot_3D():

    matplotlib.rcParams.update({'font.size': 10})

    for num in range(0, 300):

        V, x1, x2, x3 = np.load(f'output/3D/data.{num:04}.npy')

        print(f'data.{num:04}.npy')

        variables = [rho, prs, vx1, vx2, vx3]
        var = ['density', 'pressure', 'velocity_x1', 'velocity_x2', 'velocity_x3']

        for i, variable in enumerate(variables):

            fig = plt.figure(figsize=(6,15))

            # Slice in the xy plane.
            ax1 = plt.subplot2grid((3, 1), (0, 0))
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.0)
            imap1 = ax1.imshow(V[variable, int(len(x3)/2.0), :, :], 
                               extent=(x1.min(), x1.max(), 
                                       x2.min(), x2.max()),
                               aspect=1,
                               origin="lower",
                               cmap=cm.viridis)
            ax1.set_xlabel('$x$')
            ax1.set_ylabel('$y$')
            cbar1 = plt.colorbar(imap1, cax=cax)
            cbar1.set_label(f'{var[i]}')

            # Slice in the xz plane.
            ax2 = plt.subplot2grid((3, 1), (1, 0))
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.0)            
            imap2 = ax2.imshow(V[variable, :, int(len(x2)/2.0), :], 
                               extent=(x1.min(), x1.max(), 
                                       x3.min(), x3.max()),
                               aspect=1,
                               origin="lower",
                               cmap=cm.viridis)
            ax2.set_xlabel('$x$')
            ax2.set_ylabel('$z$')
            cbar2 = plt.colorbar(imap2, cax=cax)
            cbar2.set_label(f'{var[i]}')

            # Slice in the zy plane.
            ax3 = plt.subplot2grid((3, 1), (2, 0))
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="5%", pad=0.0)            
            imap3 = ax3.imshow(V[variable, :, :, int(len(x1)/2.0)], 
                               extent=(x2.min(), x2.max(), 
                                       x3.min(), x3.max()),
                               aspect=1,
                               origin="lower",
                               cmap=cm.viridis)
            ax3.set_xlabel('$y$')
            ax3.set_ylabel('$z$')
            cbar3 = plt.colorbar(imap3, cax=cax)
            cbar3.set_label(f'{var[i]}')

            plt.tight_layout()
            plt.savefig(f'output/plots/3D/{var[i]}/grid_{var[i]}_{num:04}.png')
            plt.close()


#mesh_plot_3D()
mesh_plot()
#line_plot()
