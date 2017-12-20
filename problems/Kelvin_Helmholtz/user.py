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
            'resolution x1':128,
            'resolution x2':128,
            'cfl':0.3,
            'initial dt':0.00001,
            'max dt increase':1.5,
            'max time':1.0,
            'plot frequency':0.1,
            'gamma':1.666666,
            'density unit':1.0,
            'length unit':1.0,
            'velocity unit':1.0,
            'riemann':'hll',
            'reconstruction':'flat',
            'limiter':'minmod',
            'time stepping':'Euler',
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
