import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import cm

def Mesh_plot(U, g, t):

    matplotlib.rcParams.update({'font.size': 10})

    plt.figure(figsize=(10,10))
    a = plt.imshow(U[0, g.ibeg():g.iend(), g.jbeg():g.jend()])

    ax = plt.gca();

    # Major ticks
    ax.set_xticks(np.arange(g.lower_bc_ibeg(), g.imax() + 1, 10));
    ax.set_yticks(np.arange(g.lower_bc_jbeg(), g.jmax() + 1, 10));

    # Labels for major ticks
    #ax.set_xticklabels(np.arange(g.lower_bc_ibeg(), g.imax() + 1, 1));
    #ax.set_yticklabels(np.arange(g.lower_bc_jbeg(), g.jmax() + 1, 1));

    # Minor ticks
    #ax.set_xticks(np.arange(-.5, g.imax() + 1 + 0.5, 1), minor=True);
    #ax.set_yticks(np.arange(-.5, g.jmax() + 1 + 0.5, 1), minor=True);

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('plots/grid_t_{:.8f}.png'.format(t))

