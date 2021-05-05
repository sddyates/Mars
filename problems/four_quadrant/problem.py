
"""
Docstring.
"""

import sys
sys.path.append('/home/simon/Work/Programs/Mars/mars/mars/')
sys.path.append('/home/simon/Work/Programs/Mars/mars/')

from mars import main_loop
from mars.settings import rho, prs, vx1, vx2
import numpy as np


class Problem:
    """
    Synopsis
    --------
    Proble class for the four quadrant.

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
            'Name': 'Four Quadrant',

            'Dimensions': '2D',

            'min': [0.0, 0.0, 0.0],
            'max': [1.0, 1.0, 1.0],
            'resolution': [1, 1024, 1024],

            'cfl': 0.3,
            'initial dt': 1.0e-6,
            'max dt increase': 1.5,
            'initial t': 0.0,
            'max time': 0.5,

            'save interval': 1.0e-2,
            'output type': ['vtk'],
            'output primitives': True,
            'print to file': False,
            'profiling': True,
            'restart file': None,

            'gamma': 1.666666,
            'density unit': 1.0,
            'length unit': 1.0,
            'velocity unit': 1.0,

            'mpi decomposition': [1, 2, 2],
            'optimisation': 'numba',
            'riemann': 'hll',
            'reconstruction': 'linear',
            'limiter': 'minmod',
            'time stepping': 'Euler',
            'method': 'hydro',

            'boundaries': ['outflow', 'outflow', 'outflow'],
            'internal boundary': False
        }

    def initialise(self, V, g):

        Y, X = np.meshgrid(g.x[0], g.x[1], indexing='ij')
        xt = 0.8
        yt = 0.8

        region_1 = (X < xt) & (Y < yt)
        region_2 = (X < xt) & (Y > yt)
        region_3 = (X > xt) & (Y < yt)
        region_4 = (X > xt) & (Y > yt)

        V[rho, region_1] = 0.138
        V[vx1, region_1] = 1.206
        V[vx2, region_1] = 1.206
        V[prs, region_1] = 0.029

        V[rho, region_2] = 0.5323
        V[vx1, region_2] = 1.206
        V[vx2, region_2] = 0.0
        V[prs, region_2] = 0.3

        V[rho, region_3] = 0.5323
        V[vx1, region_3] = 0.0
        V[vx2, region_3] = 1.206
        V[prs, region_3] = 0.3

        V[rho, region_4] = 1.5
        V[vx1, region_4] = 0.0
        V[vx2, region_4] = 0.0
        V[prs, region_4] = 1.5

    def internal_bc():
        return None


if __name__ == "__main__":
    main_loop(Problem())
