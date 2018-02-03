import numpy as np

def numpy_dump(V, g, num):

    np.save(f'output/1D/data.{num:04}.npy', (V, g.x1))
    np.save(f'output/2D/data.{num:04}.npy', (V, g.x1, g.x2))
    np.save(f'output/3D/data.{num:04}.npy', (V, g.x1, g.x2, g.x3))