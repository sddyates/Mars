
from mars import main_loop
from mars.settings import *
import numpy as np
import sys

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
            'Name':'Shock Cloud',

            'Dimensions': '2D',

            'min': [0.0, 0.0, 0.0],
            'max': [1.0, 1.0, 1.0],
            'resolution': [1, 768, 768],

            'cfl': 0.3,
            'initial dt': 1.0e-6,
            'max dt increase': 1.5,
            'initial t': 0.0,
            'max time': 1.0e-2,

            'save frequency': 1.0e-2,
            'output type': ['vtk'],
            'output primitives': True,
            'print to file': False,
            'profiling': True,
            'restart file': None,

            'gamma': 1.666666,
            'density unit': 1.0,
            'length unit': 1.0,
            'velocity unit': 1.0,

            'mpi decomposition': [1, 2, 3],
            'optimisation': 'numba',
            'riemann': 'hllc',
            'reconstruction': 'linear',
            'limiter': 'minmod',
            'time stepping': 'RK2',
            'method': 'hydro',

            'boundaries': ['outflow', 'outflow', 'outflow'],
            'internal boundary': False
        }

    def initialise(self, V, g):

        x_shift = 0.8
        y_shift = 0.5
        z_shift = 0.5

        if self.parameter['Dimensions'] == '2D':
            Y, X = np.meshgrid(g.x[0], g.x[1], indexing='ij')
            R = np.sqrt((X - x_shift)**2 + (Y - y_shift)**2)

        if self.parameter['Dimensions'] == '3D':
            Z, Y, X = np.meshgrid(g.x[0], g.x[1], g.x[2], indexing='ij')
            R = np.sqrt((X - x_shift)**2 + (Y - y_shift)**2 + (Z - z_shift)**2)

        shock = 0.6
        V[rho, X < shock] = 3.86859
        V[prs, X < shock] = 167.345
        V[vx1, X < shock] = 0.0
        V[vx2, X < shock] = 0.0

        V[rho, X > shock] = 1.0
        V[prs, X > shock] = 1.0
        V[vx1, X > shock] = -11.2536
        V[vx2, X > shock] = 0.0

        cloud = 0.15
        V[rho, R < cloud] = 10.0

    def internal_bc():
        return None


if __name__ == "__main__":
    main_loop(Problem())
