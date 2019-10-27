
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
        Set all variables in each cell to initialise the simulation.

    internal_bc
        Specify the internal boundary for the simulation.

    TODO
    ----
    None
    """

    def __init__(self):
        self.parameter = {
            'Name':'Shock Cloud',

            'Dimensions':'3D',
            'x1 min':0.0,
            'x1 max':1.0,
            'x2 min':0.0,
            'x2 max':1.0,
            'x3 min':0.0,
            'x3 max':1.0,

            'resolution x1':128,
            'resolution x2':128,
            'resolution x3':128,

            'cfl':0.33,
            'initial dt':1.0e-4,
            'max dt increase':1.0,
            'max time':1.0e-1,

            'plot frequency': 1.0e-2,
            'print to file':False,
            'profiling': True,

            'gamma':1.666666,
            'density unit':1.0,
            'length unit':1.0,
            'velocity unit':1.0,

            'riemann':'tvdlf',
            'reconstruction':'linear',
            'limiter':'minmod',
            'time stepping':'RK2',
            'method':'hydro',

            'lower x1 boundary':'outflow',
            'lower x2 boundary':'outflow',
            'lower x3 boundary':'outflow',
            'upper x1 boundary':'outflow',
            'upper x2 boundary':'outflow',
            'upper x3 boundary':'outflow',

            'internal boundary':False
            }

    def initialise(self, V, g, l):

        if self.parameter['Dimensions'] == '2D':
            Y, X = np.meshgrid(g.x1, g.x2, indexing='ij')
            R = np.sqrt((X - 0.8)**2 + (Y - 0.5)**2)

        if self.parameter['Dimensions'] == '3D':
            Z, Y, X = np.meshgrid(g.x1, g.x2, g.x3, indexing='ij')
            R = np.sqrt((X - 0.8)**2 + (Y - 0.5)**2 + (Z - 0.5)**2)

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
