
import os
import sys
import numpy as np
from evtk.hl import gridToVTK, imageToVTK
import h5py

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


    def __init__(self, p, l):
        self.save_freq = np.float64(p['save frequency'])
        self.output_number = 0
        self.output_prims = p['output primitives']
        self._io_type = p['output type']
        self._recognised_types = [
            'numpy',
            'vtk',
            'h5'
        ]

        self.write_file = []
        if 'numpy' in self._io_type:
            self.write_file.append(self._write_numpy)

        if ('vtk' in self._io_type) & (p['Dimensions'] != '1D'):
            self.write_file.append(self._write_vtk)

        if 'h5' in self._io_type:
            self.write_file.append(self._write_h5)

        if any(~np.isin(self._io_type, self._recognised_types)):
            print(f'Error: {self._io_type} invalid output format(s).')
            sys.exit()


    #def output(self, U, g, a, p):
    #    if self.output_number == 0:
    #        dump(U, g, a, p, self.output_number)
    #        self.output_number += 1

    #    if (self.save_freq > 0.0) \
    #        & ((g.t) >= self.output_number*self.save_freq):
    #        dump(U, g, a, p, self.output_number)
    #        self.output_number += 1


    def output(self, U, g, a, p):
        if (self.save_freq > 0.0) \
            & (g.t >= self.output_number*self.save_freq):
            print(f"    Writing {self._io_type}: {self.output_number:04}")

            if self.output_prims:
                V = self._convert(U, a.gamma_1)
            for write in self.write_file:
                write(V, g, a, p, self.output_number)
                self.output_number += 1

        return


    def _convert(self, U, gamma_1):
        V = np.zeros(shape=U.shape, dtype=np.float64)
        cons_to_prims(U, V, gamma_1)
        del U
        return V


    def _write_h5(self):
        file = h5py.File(f"output/2D/data.{self.output_number:04}.h5", "w")


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


    def _write_numpy(self, V, g, a, p, num):
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
