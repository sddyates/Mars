from mars import main_loop
from mars.settings import *
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
            'x1 min': 0.0,
            'x1 max': 4.0*np.pi,
            'x2 min': 0.0,
            'x2 max': 1.0,
            'x3 min': 0.0,
            'x3 max': 1.0,

            'resolution x1': 1024,
            'resolution x2': 128,
            'resolution x3': 128,

            'cfl': 0.6,
            'initial dt': 1.0e-4,
            'max dt increase': 1.0,
            'max time': 1.0,

            'plot frequency': 1.0e-1,
            'print to file': False,
            'profiling': True,

            'gamma': 1.666666,
            'density unit': 1.0,
            'length unit': 1.0,
            'velocity unit': 1.0,

            'optimisation': 'numba',
            'riemann': 'hllc',
            'reconstruction': 'flat',
            'limiter': 'minmod',
            'time stepping': 'RK2',
            'method': 'hydro',

            'lower x1 boundary': 'reciprocal',
            'lower x2 boundary': 'outflow',
            'lower x3 boundary': 'outflow',
            'upper x1 boundary': 'reciprocal',
            'upper x2 boundary': 'outflow',
            'upper x3 boundary': 'outflow',

            'internal boundary': False
        }

    def initialise(self, V, g, l):

        if self.parameter['Dimensions'] == '1D':
            X = g.x1

        if self.parameter['Dimensions'] == '2D':
            Y, X = np.meshgrid(g.x1, g.x2, indexing='ij')
            R = np.sqrt((X - 0.8)**2 + (Y - 0.5)**2)

        if self.parameter['Dimensions'] == '3D':
            Z, Y, X = np.meshgrid(g.x1, g.x2, g.x3, indexing='ij')
            R = np.sqrt((X - 0.8)**2 + (Y - 0.5)**2 + Z**2)

        V[prs, :] = 2.0
        V[vx1, :] = 4.0*np.pi
        V[rho, :] = np.sin(X) + 4.0

        return

    def internal_bc():
        return None


if __name__ == "__main__":

    p = Problem()
    main_loop(p)
    '''
    resol = np.logspace(np.log10(8), np.log10(1024), 10, dtype=np.int32, endpoint=True)
    #resol = np.linspace(8, 1024, 10, dtype=np.int32, endpoint=True)
    error1 = []
    error2 = []
    for (recon, error) in zip(['linear', 'flat'], [error1, error2]):
        p.parameter['reconstruction'] = recon
        for res in resol:
            p.parameter['resolution x1'] = res
            error.append(main_loop(p))

    f, ax1 = plt.subplots()
    ax1.loglog(resol, error1, 'C0o-', label='linear')
    ax1.loglog(resol, error2, 'C1o-', label='flat')
    ax1.set_xlabel(r'Resolution')
    ax1.set_ylabel(r'Error')
    #ax1.set_yscale('log')
    #ax1.set_ylim(1.0e-2, 1.0e+1)
    plt.legend()
    plt.savefig(f'output/error_sin.png')
    plt.close()
    '''
