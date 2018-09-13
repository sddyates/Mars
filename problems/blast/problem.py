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
            'Name':'Spherical blast wave.',

            'Dimensions':'2D',
            'x1 min':-0.5,
            'x1 max':0.5,
            'x2 min':-0.5,
            'x2 max':0.5,
            'x3 min':-0.5,
            'x3 max':0.5,

            'resolution x1':200,
            'resolution x2':200,
            'resolution x3':200,

            'cfl':0.3,
            'initial dt':1.0e-4,
            'max dt increase':1.5,
            'max time':3.0e-1,

            'plot frequency':3.0e-2,
            'print to file':False,

            'gamma':1.666666,
            'density unit':100.0,
            'length unit':1.0,
            'velocity unit':1.0,

            'riemann':'hllc',
            'reconstruction':'linear',
            'limiter':'minmod',
            'time stepping':'RK2',
            'method':'hydro',

            'lower x1 boundary':'outflow',
            'upper x1 boundary':'outflow',
            'lower x2 boundary':'outflow',
            'upper x2 boundary':'outflow',
            'lower x3 boundary':'outflow',
            'upper x3 boundary':'outflow',

            'internal boundary':False
            }

    def initialise(self, V, g):

        if self.parameter['Dimensions'] == '2D':
            for j in range(g.jbeg, g.jend):
                for i in range(g.ibeg, g.iend):

                    V[rho, j, i] = 1.0
                    V[prs, j, i] = 1.0/self.parameter['gamma']

                    r = np.sqrt(g.x1[i]*g.x1[i] + g.x2[j]*g.x2[j])
                    if r < 0.1:
                        V[rho, j, i] = 100.0
                        V[prs, j, i] = 100.0

                    V[vx1, j, i] = 0.0
                    V[vx2, j, i] = 0.0


        if self.parameter['Dimensions'] == '3D':
            for k in range(g.kbeg, g.kend):
                for j in range(g.jbeg, g.jend):
                    for i in range(g.ibeg, g.iend):

                        V[rho, k, j, i] = 1.0
                        V[prs, k, j, i] = 1.0/self.parameter['gamma']

                        r = np.sqrt(g.x1[i]*g.x1[i] + g.x2[j]*g.x2[j] + g.x3[k]*g.x3[k])
                        if r < 0.1:
                            V[rho, k, j, i] = 100.0
                            V[prs, k, j, i] = 100.0

                        V[vx1, k, j, i] = 0.0
                        V[vx2, k, j, i] = 0.0
                        V[vx3, k, j, i] = 0.0


    def internal_bc():
        return None

if __name__ == "__main__":
    main_loop(Problem())
