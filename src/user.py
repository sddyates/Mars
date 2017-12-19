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
            'gamma':1.4,
            'density unit':1.0,
            'length unit':1.0,
            'velocity unit':1.0,
            'riemann':'tvdlf',
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
        '''
        if self.p['Dimensions'] == '2D':
            for i in range(g.ibeg, g.iend):
                for j in range(g.jbeg, g.jend):
                    if g.x1[i] < 1.0:
                        V[rho, j, i] = 1.0
                        V[prs, j, i] = 1.0
                        V[vx1, j, i] = 0.0
                        V[vx2, j, i] = 0.0
                    else:
                        V[rho, j, i] = 0.125
                        V[prs, j, i] = 0.1
                        V[vx1, j, i] = 0.0
                        V[vx2, j, i] = 0.0
        

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

        '''
        '''
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
        '''
 
    def internal_bc():
        return None
