import numpy as np
from globe import *

class User:

    def __init__(self):
        self.p = {
            'Dimensions':'2D',
            'x1 min':-0.5,
            'x2 min':-0.5,
            'x1 max':0.5,
            'x2 max':0.5,

            'resolution x1':512,
            'resolution x2':512,

            'cfl':0.3,
            'initial dt':0.00001,
            'max dt increase':1.5,
            'max time':5.0,

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
            'lower x2 boundary':'reciprocal',
            'upper x1 boundary':'reciprocal',
            'upper x2 boundary':'reciprocal',
            'internal boundary':False
            }

    def initialise(self, V, g):
        
        if self.p['Dimensions'] == '2D':
            for i in range(g.ibeg, g.iend):
                for j in range(g.jbeg, g.jend):
                    
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
 
    def internal_bc():
        return None
