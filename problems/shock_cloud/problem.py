
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
            'Name':'Shock Cloud',

            'Dimensions':'2D',
            'x1 min':0.0,
            'x1 max':1.0,
            'x2 min':0.0,
            'x2 max':1.0,
            'x3 min':0.0,
            'x3 max':1.0,

            'resolution x1':128,
            'resolution x2':128,
            'resolution x3':128,

            'cfl':0.3,
            'initial dt':1.0e-6,
            'max dt increase':1.5,
            'initial t': 0.0,
            'max time':2.0e-1,

            'save frequency': 1.0e-2,
            'output type': ['vtk'],
            'output primitives': True,
            'print to file':False,
            'profiling': True,
            'restart file':None,

            'gamma':1.666666,
            'density unit':1.0,
            'length unit':1.0,
            'velocity unit':1.0,

            'MPI': False,
            'mpi_decomp': [1, 1, 1],
            'optimisation': 'numba',
            'riemann':'hll',
            'reconstruction':'linear',
            'limiter':'minmod',
            'time stepping':'RK2',
            'method':'hydro',

            'boundaries': ['outflow', 'periodic', 'outflow'],

            'internal boundary':False
        }

    def initialise(self, V, g):

        x_shift = 0.8
        y_shift = 0.5
        z_shift = 0.5

        if self.parameter['Dimensions'] == '2D':
            Y, X = np.meshgrid(g.x2, g.x1, indexing='ij')
            R = np.sqrt((X - x_shift)**2 + (Y - y_shift)**2)

        if self.parameter['Dimensions'] == '3D':
            Z, Y, X = np.meshgrid(g.x3, g.x2, g.x1, indexing='ij')
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
