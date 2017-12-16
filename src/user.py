import numpy as np
from globe import *

class User:

    def __init__(self):
        self.p = {
            'Dimensions':'1D',
            'x1 min':0.0,
            'x2 min':0.0,
            'x1 max':1.0,
            'x2 max':1.0,
            'resolution x1':128,
            'resolution x2':32,
            'cfl':0.3,
            'initial dt':0.0001,
            'max dt increase':1.5,
            'max time':1.0,
            'plot frequency':0.1,
            'gamma':5.0/3.0,
            'density unit':1.0,
            'length unit':1.0,
            'velocity unit':1.0,
            'riemann':'hll',
            'reconstruction':'linear',
            'limiter':'minmod',
            'time stepping':'RK2',
            'method':'hydro',
            'lower x1 boundary':'outflow',
            'lower x2 boundary':'reciprocal',
            'upper x1 boundary':'outflow',
            'upper x2 boundary':'reciprocal',
            'internal boundary':False
            }

    def initialise(self, U, g):

        if self.p['Dimensions'] == '1D':
            for i in range(g.ibeg, g.iend):
                if g.x1[i] < 0.5:
                    U[rho, i] = 1.0
                    U[prs, i] = 1.0
                    U[vx1, i] = 0.0
                else:
                    U[rho, i] = 0.125
                    U[prs, i] = 0.1
                    U[vx1, i] = 0.0                    

        if self.p['Dimensions'] == '2D':
            for i in range(g.ibeg, g.iend):
                for j in range(g.jbeg, g.jend):
                    if g.x1[i] < 0.5:
                        U[rho, j, i] = 1.0
                        U[prs, j, i] = 1.0
                        U[vx1, j, i] = 0.0
                        U[vx2, j, i] = 0.0
                    else:
                        U[rho, j, i] = 0.125
                        U[prs, j, i] = 0.1
                        U[vx1, j, i] = 0.0
                        U[vx2, j, i] = 0.0
    
    def internal_bc():
        return None
