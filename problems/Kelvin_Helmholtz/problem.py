
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
            'Name':'Kelvin Helmholtz instability.',

            'Dimensions':'2D',
            'x1 min':-0.5,
            'x1 max':0.5,
            'x2 min':-0.5,
            'x2 max':0.5,
            'x3 min':-0.5,
            'x3 max':0.5,

            'resolution x1':256,
            'resolution x2':256,
            'resolution x3':0,

            'cfl':0.3,
            'initial dt':1.0e-5,
            'max dt increase':1.5,
            'initial t': 0.0,
            'max time': 5.0,

            'save frequency': 2.5e-2,
            'output type': ['numpy'],
            'output primitives': True,
            'print to file':False,
            'profiling': True,
            'restart file':None,

            'gamma':1.4,
            'density unit':1.0,
            'length unit':1.0,
            'velocity unit':1.0,

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


    def initialise(self, V, g, l):

        if self.parameter['Dimensions'] == '2D':
            Y, X = np.meshgrid(g.x1, g.x2, indexing='ij')

        if self.parameter['Dimensions'] == '3D':
            Z, Y, X = np.meshgrid(g.x1, g.x2, g.x3, indexing='ij')

        yp = 0.25
        dens_1 = 2.0
        dens_2 = 1.0
        pres = 2.0
        vel_1 = 0.5
        vel_2 = 0.0
        amp = 0.001

        vx1_per = (np.random.random(V.shape)*2.0 - 1)*amp
        vx2_per = (np.random.random(V.shape)*2.0 - 1)*amp

        region_1 = np.absolute(Y) < yp
        region_2 = np.absolute(Y) > yp

        V[rho, region_1] = dens_1
        V[prs, region_1] = pres
        V[vx1, region_1] = vel_1 + vx1_per[vx1, region_1]
        V[vx2, region_1] = vel_2 + vx2_per[vx2, region_1]

        V[rho, region_2] = dens_2
        V[prs, region_2] = pres
        V[vx1, region_2] = -vel_1 + vx1_per[vx1, region_2]
        V[vx2, region_2] = vel_2 + vx2_per[vx2, region_2]


    def internal_bc(self):
        return None


if __name__ == "__main__":
    main_loop(Problem())
