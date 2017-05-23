import numpy as np

def parameters():
    return {'xmin':-0.5,
            'ymin':-0.5,
            'xmax':0.5,
            'ymax':0.5,
            'nx':32,
            'ny':32,
            'cfl':0.01,
            't_max':1.0,
            'gamma':5.0/3.0,
            'density_unit':1.0,
            'length_unit':1.0,
            'velocity_unit':1.0,
            'riemann':'hll',
            'reconstruction':'linear',
            'limiter':None,
            'time stepping':'RK2',
            'method':'hydro',
            'lower_bc_x':'reciprocal',
            'lower_bc_y':'reciprocal',
            'upper_bc_x':'reciprocal',
            'upper_bc_y':'reciprocal',
            'internal_bc':False
            }

def initialise(U, g):

    rho = 0
    prs = 1
    vx1 = 2
    vx2 = 3
    x = g.x()
    y = g.y()
    ibeg = g.ibeg()
    jbeg = g.jbeg()    
    iend = g.iend()
    jend = g.jend()

    for i in range(ibeg, iend):
        for j in range(jbeg, jend):
            r = np.sqrt(x[i]*x[i] + y[j]*y[j])
            if abs(x[i]) < 0.25:
                U[rho, i, j] = 2.0
                U[prs, i, j] = 2.5
                U[vx1, i, j] = 0.0 + (np.random.rand()*2.0 - 1)*0.005
                U[vx2, i, j] = 0.5 + (np.random.rand()*2.0 - 1)*0.005
            else:
                U[rho, i, j] = 1.0
                U[prs, i, j] = 2.5
                U[vx1, i, j] = 0.0 + (np.random.rand()*2.0 - 1)*0.005
                U[vx2, i, j] = -0.5 + (np.random.rand()*2.0 - 1)*0.005
    
def internal_bc():
   return None
