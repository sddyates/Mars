import sys
sys.path.insert(0, '../../src')
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpi4py import MPI
from time import time

def line_plot():

    for num in range(0, 11):

        f, ax1 = plt.subplots()

        x1p, x2p, Prho, Pvx1, Pprs = np.loadtxt(f'output/pluto/data.{num:04}.tab', unpack=True)
        V, x1 = np.load(f'output/1D/data_1D_{num:04}.npy')

        ax1.plot(x1, V[rho], 'b')
        ax1.plot(x1, V[prs], 'r')
        ax1.plot(x1, V[vx1], 'g')
        ax1.plot(x1p, Prho, 'b--')
        ax1.plot(x1p, Pprs, 'r--')
        ax1.plot(x1p, Pvx1, 'g--')

        if not os.path.isdir(f'output/plots/'):
            os.makedirs(f'output/plots/')

        plt.savefig(f'output/plots/line_{num:04}.png')
        plt.close()

def mesh_plot():

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    matplotlib.rcParams.update({'font.size': 10})

    # if rank == 0:
    #     start_time = time()

    if os.path.isdir(f'output/plots/2D/density'):
        plot_folder = f"output/plots/2D/density/"
        plot_list = [plot_folder+f for f in os.listdir(plot_folder) if f.endswith('.png')]
        plot_list.sort()
        plot_number = len(plot_list)
    else:
        plot_number = 0

    folder = f"output/2D/"
    file_list = [folder+f for f in os.listdir(folder) if f.endswith('.npy')]
    file_list.sort()
    file_number = len(file_list)

        # if file_number%size == 0:
        #     # integer_divisable =
        #     files_per_rank = int(file_number/size)
        # else:
        #     files_per_rank = file_number//size + 1#int((file_number - file_number%size)/(size - 1))
        #
        # perrank = len(file_list)//size
        #
        # list_of_file_lists = [
        #     file_list[i:i + files_per_rank] \
        #     for i in range(0, file_number, files_per_rank)
        # ]

    # else:
    #     list_of_file_lists = None

    # Scatter the devided list to processes.
    # local_file_list = comm.scatter(list_of_file_lists, root=0)


    matplotlib.rcParams.update({'font.size': 10})

    for num, file in enumerate(file_list[plot_number:]):
    #for num in range(47, 300):

        num += plot_number

        #V, x1, x2 = np.load(f'output/2D/data_2D_{num:04}.npy', allow_pickle=True)
        V, x1, x2 = np.load(file, allow_pickle=True)

        print(f"Processing file: {file}")

        variables = [
            V[0, :, :],
            V[1, :, :],
            V[2, :, :],
            V[3, :, :]
        ]
        var = [
            'density',
            'pressure',
            'velocity_x1',
            'velocity_x2'
        ]

        for i, variable in enumerate(variables):

            plt.figure(figsize=(10,10))
            a = plt.imshow(variable, extent=(x1.min(), x1.max(),
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

            if not os.path.isdir(f'output/plots/2D//{var[i]}'):
                os.makedirs(f'output/plots/2D//{var[i]}')

            plt.savefig(f'output/plots/2D/{var[i]}/grid_{var[i]}_{num:04}.png')
            plt.close()


mesh_plot()
#line_plot()
