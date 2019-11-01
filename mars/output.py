
import os
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


class OutputInput:


    def __init__(self, p):
        self.save_freq = np.float64(p['save frequency'])
        self.output_number = 0

        self.write_file = []
        if p['output type'].any() == 'numpy':
            self.write_file.append(self._write_numpy)
        elif (p['output type'].any() == 'vtk') & (p['Dimensions'] != '1D'):
            self.write_file.append(self._write_vtk)
        elif p['output type'].any() == 'h5':
            self.write_file.append(self._write_h5)
        else:
            print('Error: invalid output format.')
            sys.exit()


    def output(self, t):
        if (self.save_freq > 0.0) & ((t + dt) > num*self.save_freq):
            timing.start_io()
            dump(U, grid, a)
            self.output_number += 1
            timing.stop_io()


    def write(self, A, g, a, p, num):

        if num == 0:
            print(f"    Writing initial conditions: 0000")
        else:
            print(f"    Writing output file: {num:04}")

        if p['output primitives']:
            A = self._convert(A, a)
        for write in self.write_file:
            write(A, g, a, p, num)

        return


        def _write_h5(self):
            None


        def _convert(self, U):
            V = np.zeros(shape=U.shape, dtype=np.float64)
            cons_to_prims(U, V, a.gamma_1)
            del U
            return V


        def _write_vtk(self, V, g, a, p, num):
            if p['Dimensions'] == '2D':
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
                imageToVTK(
                    f"output/2D/data.{self.output_number:04}",
                    origin = (g.x1[g.ibeg], g.x2[g.jbeg], 0.0),
                    spacing = (g.dx1, g.dx2, 0.0),
                    cellData = {"rho":V_vtk_rho,
                                "prs":V_vtk_prs,
                                "vx1":V_vtk_vx1,
                                "vx2":V_vtk_vx2}
                )

            if p['Dimensions'] == '3D':
                gridToVTK(
                    f"output/3D/data.{self.output_number:04}",
                    g.x1_verts,
                    g.x2_verts,
                    g.x3_verts,
                    cellData = {"rho":V[rho].T,
                                "prs":V[prs].T,
                                "vx1":V[vx1].T,
                                "vx2":V[vx2].T,
                                "vx3":V[vx3].T}
                )


        def _write_numpy(self):
            if p['Dimensions'] == '1D':
                np.save(
                    f'output/1D/data.{self.output_number:04}.npy',
                    (V, g.x1)
                )
            if p['Dimensions'] == '2D':
                np.save(
                    f'output/2D/data.{self.output_number:04}.npy',
                    (V, g.x1, g.x2)
                )
            if p['Dimensions'] == '3D':
                np.save(
                    f'output/3D/data.{self.output_number:04}.npy',
                    (V, g.x1, g.x2, g.x3)
                )


    def read(self):
        None


        def _read_h5(self):
            None


        def _read_vtk(self):
            None


        def _read_numpy(self):
            None
