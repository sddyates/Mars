
from mars import main_loop
import numpy as np
from mars.settings import *

class Problem:
    """
    Synopsis
    --------
    User class for the Kelvin-Helmholtz instability

    Args
    ----
    None

    Methods
    -------
    initialise:
        Set all variables in each cell to initialise the simulation.

    internal_bc:
        Specify the internal boundary for the simulation.

    TODO
    ----
    None
    """

    def __init__(self):
        self.parameter = {
            'Name':'Kelvin Helmholtz instability.',

            'Dimensions':'3D',
            'x1 min':-0.5,
            'x1 max':0.5,
            'x2 min':-0.5,
            'x2 max':0.5,
            'x3 min':-0.5,
            'x3 max':0.5,

            'resolution x1':128,
            'resolution x2':128,
            'resolution x3':128,

            'cfl':0.3,
            'initial dt':1.0e-5,
            'max dt increase':1.5,
            'initial time': 0.0,
            'max time': 10.0,

            'save frequency': 1.0,
            'output type': ['numpy', 'vtk', 'h5'],
            'output primitives': True,
            'print to file':False,
            'profiling': True,
            'restart file': None,

            'gamma':1.4,
            'density unit': 1.0,
            'length unit': 1.0,
            'velocity unit': 1.0,
            'normalise': True,

            'optimisation': 'numba',
            'riemann':'hllc',
            'reconstruction':'linear',
            'limiter':'minmod',
            'time stepping':'RK2',
            'method':'hydro',

            'lower x1 boundary':'reciprocal',
            'upper x1 boundary':'reciprocal',
            'lower x2 boundary':'reciprocal',
            'upper x2 boundary':'reciprocal',
            'lower x3 boundary':'reciprocal',
            'upper x3 boundary':'reciprocal',

            'internal boundary':False
            }


    def initialise(self, V, grid):

        if self.parameter['Dimensions'] == '2D':
            Y, X = np.meshgrid(grid.x2, grid.x1, indexing='ij')

        if self.parameter['Dimensions'] == '3D':
            Z, Y, X = np.meshgrid(grid.x3, grid.x2, grid.x1, indexing='ij')

        boundary = 0.25
        amplitude = 0.001
        inner = np.absolute(Y) < boundary
        outer = np.absolute(Y) > boundary
        random = np.random.random_sample

        V[prs] = 2.5

        V[rho, inner] = 2.0
        V[vx1, inner] = 0.5 + (random(V[vx1, inner].shape)*2.0 - 1)*amplitude
        V[vx2, inner] = (random(V[vx2, inner].shape)*2.0 - 1)*amplitude
        V[vx3, inner] = (random(V[vx3, inner].shape)*2.0 - 1)*amplitude

        V[rho, outer] = 1.0
        V[vx1, outer] = -0.5 + (random(V[vx1, outer].shape)*2.0 - 1)*amplitude
        V[vx2, outer] = (random(V[vx2, outer].shape)*2.0 - 1)*amplitude
        V[vx3, outer] = (random(V[vx3, outer].shape)*2.0 - 1)*amplitude

        # if self.parameter['Dimensions'] == '2D':
        #     for j in range(grid.jbeg, grid.jend):
        #         for i in range(grid.ibeg, grid.iend):
        #             if abs(grid.x2[j]) < 0.25:
        #                 V[rho, j, i] = 2.0
        #                 V[prs, j, i] = 2.5
        #                 V[vx1, j, i] = 0.5 + (np.random.rand()*2.0 - 1)*0.001
        #                 V[vx2, j, i] = 0.0 + (np.random.rand()*2.0 - 1)*0.001
        #             else:
        #                 V[rho, j, i] = 1.0
        #                 V[prs, j, i] = 2.5
        #                 V[vx1, j, i] = -0.5 + (np.random.rand()*2.0 - 1)*0.001
        #                 V[vx2, j, i] = 0.0 + (np.random.rand()*2.0 - 1)*0.001
        #
        # if self.parameter['Dimensions'] == '3D':
        #     for k in range(grid.kbeg, grid.kend):
        #         for j in range(grid.jbeg, grid.jend):
        #             for i in range(grid.ibeg, grid.iend):
        #                 if abs(grid.x2[j]) < 0.25:
        #                     V[rho, k, j, i] = 2.0
        #                     V[prs, k, j, i] = 2.5
        #                     V[vx1, k, j, i] = 0.5 + (np.random.rand()*2.0 - 1)*0.001
        #                     V[vx2, k, j, i] = 0.0 + (np.random.rand()*2.0 - 1)*0.001
        #                     V[vx3, k, j, i] = 0.0 + (np.random.rand()*2.0 - 1)*0.001
        #                 else:
        #                     V[rho, k, j, i] = 1.0
        #                     V[prs, k, j, i] = 2.5
        #                     V[vx1, k, j, i] = -0.5 + (np.random.rand()*2.0 - 1)*0.001
        #                     V[vx2, k, j, i] = 0.0 + (np.random.rand()*2.0 - 1)*0.001
        #                     V[vx3, k, j, i] = 0.0 + (np.random.rand()*2.0 - 1)*0.001


    def source(self, U, grid):



        return U


if __name__ == "__main__":
    main_loop(Problem())
