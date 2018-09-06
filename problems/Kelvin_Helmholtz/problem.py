
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
        Set all variables in each cell to  initialise the simulation.

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

            'resolution x1':200,
            'resolution x2':200,
            'resolution x3':32,

            'cfl':0.3,
            'initial dt':1.0e-5,
            'max dt increase':1.5,
            'max time':15.0,

            'plot frequency':0.1,
            'print to file':False,

            'gamma':1.4,
            'density unit':1.0,
            'length unit':1.0,
            'velocity unit':1.0,

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


    def initialise(self, V, g):
        
        if self.parameter['Dimensions'] == '2D':
            for j in range(g.jbeg, g.jend):
                for i in range(g.ibeg, g.iend):                    
                    if abs(g.x2[j]) < 0.25:
                        V[rho, j, i] = 2.0
                        V[prs, j, i] = 2.5
                        V[vx1, j, i] = 0.5 + (np.random.rand()*2.0 - 1)*0.001
                        V[vx2, j, i] = 0.0 + (np.random.rand()*2.0 - 1)*0.001
                    else:
                        V[rho, j, i] = 1.0
                        V[prs, j, i] = 2.5
                        V[vx1, j, i] = -0.5 + (np.random.rand()*2.0 - 1)*0.001
                        V[vx2, j, i] = 0.0 + (np.random.rand()*2.0 - 1)*0.001
 
        if self.parameter['Dimensions'] == '3D':
            for k in range(g.kbeg, g.kend):
                for j in range(g.jbeg, g.jend):
                    for i in range(g.ibeg, g.iend):                    
                        if abs(g.x2[j]) < 0.25:
                            V[rho, k, j, i] = 2.0
                            V[prs, k, j, i] = 2.5
                            V[vx1, k, j, i] = 0.5 + (np.random.rand()*2.0 - 1)*0.001
                            V[vx2, k, j, i] = 0.0 + (np.random.rand()*2.0 - 1)*0.001
                            V[vx3, k, j, i] = 0.0 + (np.random.rand()*2.0 - 1)*0.001
                        else:
                            V[rho, k, j, i] = 1.0
                            V[prs, k, j, i] = 2.5
                            V[vx1, k, j, i] = -0.5 + (np.random.rand()*2.0 - 1)*0.001
                            V[vx2, k, j, i] = 0.0 + (np.random.rand()*2.0 - 1)*0.001
                            V[vx3, k, j, i] = 0.0 + (np.random.rand()*2.0 - 1)*0.001


    def internal_bc(self):
        return None


if __name__ == "__main__":
    main_loop(Problem())



