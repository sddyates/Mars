import numpy as np
from globe import *

class User:

    def __init__(self):
        self.p = {
            'Dimensions':'2D',
            'x1 min':0.0,
            'x2 min':0.0,
            'x1 max':1.0,
            'x2 max':1.0,
            'resolution x1':128,
            'resolution x2':128,
            'cfl':0.3,
            'initial dt':0.000001,
            'max dt increase':1.5,
            'max time':0.1,
            'plot frequency':0.001,
            'gamma':1.666666,
            'density unit':1.0,
            'length unit':1.0,
            'velocity unit':1.0,
            'riemann':'hll',
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

    def initialise(self, V, g):

        if self.p['Dimensions'] == '2D':
            for i in range(g.ibeg, g.iend):
                for j in range(g.jbeg, g.jend):
                    
                    xp = g.x1[i] - 0.8
                    yp = g.x2[j] - 0.5
                    r = np.sqrt(xp*xp + yp*yp)
                    if (g.x1[i] < 0.6):
                        V[rho, j, i] = 3.86859
                        V[prs, j, i] = 167.345
                        V[vx1, j, i] = 0.0
                        V[vx2, j, i] = 0.0
                    if (g.x1[i] > 0.6):
                        V[rho, j, i] = 1.0
                        V[prs, j, i] = 1.0
                        V[vx1, j, i] = -11.2536
                        V[vx2, j, i] = 0.0
                    if r < 0.15:
                        V[rho, j, i] = 10.0

    def internal_bc():
        return None
