from mars import main_loop
from mars.settings import *
import numpy as np

class Problem:
    """
    Synopsis
    --------
    Problem class for the shock tube.

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
            'Name':'Shock Tube',

            'Dimensions':'1D',
            'x1 min':0.0,
            'x1 max':1.0,

            'resolution x1':128,

            'cfl':0.3,
            'initial dt':1.0e-5,
            'max dt increase':1.5,
            'max time':1.0,

            'plot frequency':1.0e-1,
            'print to file':False,

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
            'upper x1 boundary':'outflow',
            'upper x2 boundary':'outflow',
            'internal boundary':False
            }

    def initialise(self, V, grid):        

        if self.parameter['Dimensions'] == '1D':
            for i in range(grid.ibeg, grid.iend):
                if grid.x1[i] < 0.5:
                    V[rho, i] = 1.0
                    V[prs, i] = 1.0
                    V[vx1, i] = 0.0
                else:
                    V[rho, i] = 0.125
                    V[prs, i] = 0.1
                    V[vx1, i] = 0.0                    
 
    #def internal_bc():
    #    return None


if __name__ == "__main__":
    main_loop(Problem())