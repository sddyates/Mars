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
            'resolution x2':1,

            'cfl':0.3,
            'initial dt':0.00001,
            'max dt increase':1.5,
            'max time':1.0,

            'plot frequency':0.1,
            'print to file':False,

            'gamma':1.666666,
            'density unit':1.0,
            'length unit':1.0,
            'velocity unit':1.0,

            'riemann':'hllc',
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
        
        if self.p['Dimensions'] == '1D':
            for i in range(g.ibeg, g.iend):
                if g.x1[i] < 0.5:
                    V[rho, i] = 1.0
                    V[prs, i] = 1.0
                    V[vx1, i] = 0.0
                else:
                    V[rho, i] = 0.125
                    V[prs, i] = 0.1
                    V[vx1, i] = 0.0                    
 
    def internal_bc():
        return None
