
import os
import sys
import numpy as np
import evtk
import h5py

from settings import *
from tools import cons_to_prims


class OutputInput:


    def __init__(self, p, l):

        self._output_number = 0
        self._save_freq = np.float64(p['save frequency'])
        self._output_prims = p['output primitives']
        self._io_type = p['output type']
        self._io_folder = "./output/" + p['Dimensions'] + "/"

        if not os.path.isdir(self._io_folder):
            os.makedirs(self._io_folder)
            print(f'    Created io folder at: "{self._io_folder}"')

        self._file_prefix = "data_" + p['Dimensions'] + "_"

        self._base_file_name = self._io_folder + self._file_prefix

        if p['restart file'] is not None:
            self._restart_file = self._io_folder + self._file_prefix + f"{p['restart file']:04}" + ".h5"

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


    def output(self, U, g, a, p):
        if (self._save_freq > 0.0) \
            & (g.t >= self._output_number*self._save_freq):

            self._file_name = self._base_file_name + f"{self._output_number:04}"

            print(f"    Writing {self._io_type}: {self._output_number:04}")

            if self._output_prims:
                V = self._convert(U, a.gamma_1)

            for write in self.write_file:
                write(V, g, a, p)

            self._output_number += 1

        return


    def _convert(self, U, gamma_1):
        V = np.zeros(shape=U.shape, dtype=np.float64)
        cons_to_prims(U, V, gamma_1)
        del U
        return V


    def _write_h5(self, V, g, a, p):

        data = h5py.File(self._file_name + ".h5", "w")

        data.create_dataset('density', data=V[rho])
        data.create_dataset('pressure', data=V[prs])
        data.create_dataset('velocity x1', data=V[vx1])
        origin = [g.x1min]
        extent = [g.x1max]
        resolution = [g.nx1]

        if p['Dimensions'] == "2D":
            data.create_dataset('velocity x2', data=V[vx2])
            origin.append(g.x2min)
            extent.append(g.x2max)
            resolution.append(g.nx2)

        if p['Dimensions'] == "3D":
            data.create_dataset('velocity x3', data=V[vx3])
            origin.append(g.x3min)
            extent.append(g.x3max)
            resolution.append(g.nx3)

        data.attrs['origin'] = origin[::-1]
        data.attrs['extent'] = extent[::-1]
        data.attrs['resolution'] = resolution[::-1]
        data.attrs['time'] = g.t
        data.attrs['dt'] = g.dt
        data.attrs['output_number'] = self._output_number

        data.close()
        return


    def _write_vtk(self, V, g, a, p):
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
            evtk.hl.imageToVTK(
                self._file_name,
                origin = (g.x1[g.ibeg], g.x2[g.jbeg], 0.0),
                spacing = (g.dx1, g.dx2, 0.0),
                cellData = {"rho":V_vtk_rho,
                            "prs":V_vtk_prs,
                            "vx1":V_vtk_vx1,
                            "vx2":V_vtk_vx2}
            )

        if p['Dimensions'] == '3D':
            evtk.hl.gridToVTK(
                self._file_name,
                g.x1_verts,
                g.x2_verts,
                g.x3_verts,
                cellData = {"rho":V[rho].T,
                            "prs":V[prs].T,
                            "vx1":V[vx1].T,
                            "vx2":V[vx2].T,
                            "vx3":V[vx3].T}
            )

        return


    def _write_numpy(self, V, g, a, p):
        if p['Dimensions'] == '1D':
            np.save(self._file_name, (V, g.x1))
        if p['Dimensions'] == '2D':
            np.save(self._file_name, (V, g.x1, g.x2))
        if p['Dimensions'] == '3D':
            np.save(self._file_name, (V, g.x1, g.x2, g.x3))
        return


    def input(self, p):
        return self._read_h5(p)


    def _read_h5(self, p):

        print(f'    Restarting from file: "{self._restart_file}"')

        data = h5py.File(self._restart_file, "r")
        array_size = [len([var for var in data])]

        density = data['density'][()]
        pressure = data['pressure'][()]
        velocity_x1 = data['velocity x1'][()]

        p['x1 min'] = data.attrs['origin'][-1]
        p['x1 max'] = data.attrs['extent'][-1]
        p['resolution x1'] = data.attrs['resolution'][-1]
        array_size.append(density.shape[-1])
        if p['Dimensions'] == "2D":
            velocity_x2 = data['velocity x2'][()]
            p['x2 min'] = data.attrs['origin'][-2]
            p['x2 max'] = data.attrs['extent'][-2]
            p['resolution x2'] = data.attrs['resolution'][-2]
            array_size.append(density.shape[-2])
        if p['Dimensions'] == "3D":
            velocity_x3 = data['velocity x3'][()]
            p['x3 min'] = data.attrs['origin'][-3]
            p['x3 max'] = data.attrs['extent'][-3]
            p['resolution x3'] = data.attrs['resolution'][-3]
            array_size.append(density.shape[-3])

        self._output_number = data.attrs['output_number']
        p['initial t'] = data.attrs['time']
        p['initial dt'] = data.attrs['dt']

        V = np.zeros(array_size, dtype=np.float64)

        V[rho] = density
        V[prs] = pressure
        V[vx1] = velocity_x1
        if p['Dimensions'] == "2D":
            V[vx2] = velocity_x2
        if p['Dimensions'] == "3D":
            V[vx3] = velocity_x3

        return V


    def _read_numpy(self):
        V, x1, x2, x3 = np.load()
        None
