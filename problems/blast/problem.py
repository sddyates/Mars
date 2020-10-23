from mars import main_loop
from mars.settings import *
import numpy as np


class Problem:
    """
    Synopsis
    --------
    User class for the shock cloud.

    Args
    ----
    None

    Methods
    -------
    initialise
        Set all variables in each cell to  initialise the simulation.

    internal_bc
        Specify the internal boundary for the simulation.

    TODO
    ----
    None
    """

    def __init__(self):
        self.parameter = {
            'Name': 'Spherical blast wave.',

            'Dimensions': '2D',

            'min': [-0.5, -0.5, -0.5],
            'max': [0.5, 0.5, 0.5],
            'resolution': [1, 512, 512],

            'cfl': 0.3,
            'initial dt': 1.0e-4,
            'max dt increase': 1.5,
            'initial t': 0.0,
            'max time': 3.0e-1,

            'save frequency': 3.0e-2,
            'output type': ['vtk'],
            'output primitives': True,
            'print to file': False,
            'profiling': True,
            'restart file': None,

            'gamma': 1.666666,
            'density unit': 100.0,
            'length unit': 1.0,
            'velocity unit': 1.0,

            'mpi decomposition': [1, 1, 1],
            'optimisation': 'numba',
            'riemann': 'tvdlf',
            'reconstruction': 'linear',
            'limiter': 'minmod',
            'time stepping': 'RK2',
            'method': 'hydro',

            'boundaries': ['outflow', 'outflow', 'outflow'],
            'internal boundary': False
            }

    def initialise(self, V, g):

        if self.parameter['Dimensions'] == '2D':
            Y, X = np.meshgrid(g.x[0], g.x[1], indexing='ij')
            R = np.sqrt(X*X + Y*Y)

        if self.parameter['Dimensions'] == '3D':
            Z, Y, X = np.meshgrid(g.x[0], g.x[1], g.x[2], indexing='ij')
            R = np.sqrt(X*X + Y*Y + Z*Z)

        V[vx1:] = 0.0

        V[rho] = 1.0
        V[prs] = 1.0/self.parameter['gamma']

        V[rho, R < 0.1] = 100.0
        V[prs, R < 0.1] = 100.0

    def internal_bc():
        return None


if __name__ == "__main__":
    main_loop(Problem())
