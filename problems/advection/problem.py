
"""
Docstring.
"""

import sys
sys.path.append('/home/simon/Work/Programs/Mars/mars/mars/')
sys.path.append('/home/simon/Work/Programs/Mars/mars/')

from mars import main_loop
from mars.settings import rho, prs, vx1, vx2, vx3
import numpy as np
import matplotlib.pyplot as plt

class Problem:
    """
    Synopsis
    --------
    User class for the advection problem.
    Args
    ----
    None
    Methods
    -------
    initialise
        Set all variables in each cell to initialise the simulation.
    internal_bc
        Specify the internal boundary for the simulation.
    TODO
    ----
    None
    """

    def __init__(self):
        self.parameter = {
            'Name': 'Advection',

            'Dimensions': '1D',

            'min': [0.0, 0.0, 0.0],
            'max': [1.0, 1.0, 4.0*np.pi],
            'resolution': [1, 1, 1024],

            'cfl': 0.6,
            'initial dt': 1.0e-4,
            'max dt increase': 1.0,
            'initial t': 0.0,
            'max time': 1.0,

            'save interval': 1.0e-1,
            'output type': ['vtk'],
            'output primitives': True,
            'print to file': False,
            'profiling': True,
            'restart file': None,

            'gamma': 1.666666,
            'density unit': 1.0,
            'length unit': 1.0,
            'velocity unit': 1.0,

            'mpi decomposition': [1, 1, 4],
            'optimisation': 'numba',
            'riemann': 'hll',
            'reconstruction': 'linear',
            'limiter': 'minmod',
            'time stepping': 'RK2',
            'method': 'hydro',

            'boundaries': ['outflow', 'outflow', 'reciprocal'],
            'internal boundary': False
        }

    def initialise(self, V, g):

        x0 = np.array(g.x[0], dtype=np.float64)

        V[prs, :] = 2.0
        V[vx1, :] = 4.0*np.pi
        V[rho, :] = np.sin(x0) + 4.0

        return

    def internal_bc():
        return None


if __name__ == "__main__":

    p = Problem()
    # main_loop(p)

    resol = np.logspace(np.log10(8), np.log10(1024), 10, dtype=np.int32, endpoint=True)
    resol = np.linspace(8, 1024, 10, dtype=np.int32, endpoint=True)
    error1 = []
    error2 = []
    for (recon, error) in zip(['linear', 'flat'], [error1, error2]):
        p.parameter['reconstruction'] = recon
        for res in resol:
            p.parameter['resolution'] = [1, 1, res]
            U = main_loop(p)
            V2 = np.zeros_like(U[rho])
            V2 = np.sin(grid.x1) + 4.0
            error.append(np.absolute((V2 - U[rho])).sum()/len(grid.x1))

    f, ax1 = plt.subplots()
    ax1.loglog(resol, error1, 'C0o-', label='linear')
    ax1.loglog(resol, error2, 'C1o-', label='flat')
    ax1.set_xlabel(r'Resolution')
    ax1.set_ylabel(r'Error')
    # ax1.set_yscale('log')
    # ax1.set_ylim(1.0e-2, 1.0e+1)
    plt.legend()
    plt.savefig('output/error_sin.png')
    plt.close()
