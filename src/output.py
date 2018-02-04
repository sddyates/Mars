import numpy as np
from globe import *
from evtk.hl import gridToVTK 

def numpy_dump(V, g, p, num):

    if p['Dimensions'] == '1D':
        np.save(f'output/1D/data.{num:04}.npy', (V, g.x1))

    if p['Dimensions'] == '2D':    
        np.save(f'output/2D/data.{num:04}.npy', (V, g.x1, g.x2))

    if p['Dimensions'] == '3D':
        np.save(f'output/3D/data.{num:04}.npy', (V, g.x1, g.x2, g.x3))

        gridToVTK(f"output/3D/data.{num:04}", 
                  g.x1_verts, g.x2_verts, g.x3_verts, 
                  cellData = {"rho":V[rho].T, 
                              "prs":V[prs].T,
                              "vx1":V[vx1].T,
                              "vx2":V[vx2].T,
                              "vx3":V[vx3].T})

