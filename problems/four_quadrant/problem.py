
from mars import main_loop
from mars.settings import *
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
            'x1 min': 0.0,
            'x1 max': 1.0,
            'x2 min': 0.0,
            'x2 max': 1.0,
            'x3 min': 0.0,
            'x3 max': 1.0,

            'resolution x1': 512,
            'resolution x2': 512,
            'resolution x3': 0,

            'cfl': 0.3,
            'initial dt': 1.0e-6,
            'max dt increase': 1.5,
            'max time': 0.5,

            'plot frequency': 5.0e-2,
            'print to file': False,

            'gamma': 1.666666,
            'density unit': 1.0,
            'length unit': 1.0,
            'velocity unit': 1.0,

            'riemann': 'hllc',
            'reconstruction': 'linear',
            'limiter': 'minmod',
            'time stepping': 'RK2',
            'method': 'hydro',

            'lower x1 boundary': 'outflow',
            'lower x2 boundary': 'outflow',
            'lower x3 boundary': 'outflow',
            'upper x1 boundary': 'outflow',
            'upper x2 boundary': 'outflow',
            'upper x3 boundary': 'outflow',

            'internal boundary': False
            }

    def initialise(self, V, g):

        xt = 0.8
        yt = 0.8

        if self.parameter['Dimensions'] == '2D':
            for j in range(g.jbeg, g.jend):
                for i in range(g.ibeg, g.iend):

                    if g.x1[i] < xt and g.x2[j] < yt:
                        V[rho, j, i] = 0.138
                        V[vx1, j, i] = 1.206
                        V[vx2, j, i] = 1.206
                        V[prs, j, i] = 0.029

                    if g.x1[i] < xt and g.x2[j] > yt:
                        V[rho, j, i] =  0.5323
                        V[vx1, j, i] = 1.206
                        V[vx2, j, i] = 0.0
                        V[prs, j, i] = 0.3

                    if g.x1[i] > xt and g.x2[j] < yt:
                        V[rho, j, i] = 0.5323
                        V[vx1, j, i] = 0.0
                        V[vx2, j, i] = 1.206
                        V[prs, j, i] = 0.3

                    if g.x1[i] > xt and g.x2[j] > yt:
                        V[rho, j, i] = 1.5
                        V[vx1, j, i] = 0.0
                        V[vx2, j, i] = 0.0
                        V[prs, j, i] = 1.5


    def internal_bc():
        return None


if __name__ == "__main__":
    main_loop(Problem())
