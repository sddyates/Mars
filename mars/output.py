
import numpy as np
from evtk.hl import gridToVTK, imageToVTK
from settings import *
from tools import cons_to_prims


def dump(U, g, a, p, num):

    V = np.zeros(shape=U.shape, dtype=np.float64)
    cons_to_prims(U, V, a.gamma_1)
    del U

    if num == 0:
        print(f"    Writing initial conditions: 0000")
    else:
        print(f"    Writing output file: {num:04}")

    if p['Dimensions'] == '1D':
        np.save(f'output/1D/data.{num:04}.npy', (V, g.x1))

    if p['Dimensions'] == '2D':
        np.save(f'output/2D/data.{num:04}.npy', (V, g.x1, g.x2))

        V_vtk = np.expand_dims(V, axis=4)
        V_vtk_rho = np.copy(
            np.swapaxes(V_vtk, 1, 2)[rho, g.jbeg:g.jend, g.ibeg:g.iend, :],
            order='F')
        V_vtk_prs = np.copy(
            np.swapaxes(V_vtk, 1, 2)[prs, g.jbeg:g.jend, g.ibeg:g.iend, :],
            order='F')
        V_vtk_vx1 = np.copy(
            np.swapaxes(V_vtk, 1, 2)[vx1, g.jbeg:g.jend, g.ibeg:g.iend, :],
            order='F')
        V_vtk_vx2 = np.copy(
            np.swapaxes(V_vtk, 1, 2)[vx2, g.jbeg:g.jend, g.ibeg:g.iend, :],
            order='F')

        imageToVTK(f"output/2D/data.{num:04}",
                  origin = (g.x1[g.ibeg], g.x2[g.jbeg], 0.0),
                  spacing = (g.dx1, g.dx2, 0.0),
                  cellData = {"rho":V_vtk_rho,
                              "prs":V_vtk_prs,
                              "vx1":V_vtk_vx1,
                              "vx2":V_vtk_vx2})

    if p['Dimensions'] == '3D':
        np.save(f'output/3D/data.{num:04}.npy', (V, g.x1, g.x2, g.x3))

        gridToVTK(f"output/3D/data.{num:04}",
                  g.x1_verts, g.x2_verts, g.x3_verts,
                  cellData = {"rho":V[rho].T,
                              "prs":V[prs].T,
                              "vx1":V[vx1].T,
                              "vx2":V[vx2].T,
                              "vx3":V[vx3].T})
