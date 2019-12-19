
import os
import sys
import numpy as np
import h5py
import evtk

from settings import *
from tools import cons_to_prims


class OutputInput:


    def __init__(self, parameter):

        self._output_number = 0
        self._save_freq = np.float64(parameter['save frequency'])
        self._output_prims = parameter['output primitives']
        self._io_type = parameter['output type']
        self._io_folder = "./output/" + parameter['Dimensions'] + "/"

        if not os.path.isdir(self._io_folder):
            os.makedirs(self._io_folder)
            print(f'    Created io folder at: "{self._io_folder}"')

        self._file_prefix = "data_" + parameter['Dimensions'] + "_"

        self._base_file_name = self._io_folder + self._file_prefix

        if parameter['restart file'] is not None:
            self._restart_file = self._io_folder + self._file_prefix + f"{parameter['restart file']:04}" + ".h5"

        self._recognised_types = [
            'numpy',
            'vtk',
            'h5'
        ]

        self.write_file = []
        if 'numpy' in self._io_type:
            self.write_file.append(self._write_numpy)

        if ('vtk' in self._io_type) & (parameter['Dimensions'] != '1D'):
            self.write_file.append(self._write_vtk)

        if 'h5' in self._io_type:
            self.write_file.append(self._write_h5)

        if any(~np.isin(self._io_type, self._recognised_types)):
            print(f'Error: {self._io_type} invalid output format(s).')
            sys.exit()


    def output(self, U, grid, algorithm, parameter):
        if (self._save_freq > 0.0) \
            & (grid.t >= self._output_number*self._save_freq):

            self._file_name = self._base_file_name + f"{self._output_number:04}"

            print(f"    Writing {self._io_type}: {self._output_number:04}")

            if self._output_prims:
                V = self._convert(U, algorithm.gamma_1)

            for write in self.write_file:
                write(V, grid, algorithm, parameter)

            self._output_number += 1

        return


    def _convert(self, U, gamma_1):
        V = np.zeros(shape=U.shape, dtype=np.float64)
        cons_to_prims(U, V, gamma_1)
        del U
        return V


    def _write_h5(self, V, grid, algorithm, parameter):

        data = h5py.File(self._file_name + ".h5", "w")

        data.create_dataset('density', data=V[rho])
        data.create_dataset('pressure', data=V[prs])
        data.create_dataset('velocity x1', data=V[vx1])

        origin = [grid.x1min]
        extent = [grid.x1max]
        resolution = [grid.nx1]
        if parameter['Dimensions'] == "2D":
            data.create_dataset('velocity x2', data=V[vx2])
            origin.append(grid.x2min)
            extent.append(grid.x2max)
            resolution.append(grid.nx2)
        if parameter['Dimensions'] == "3D":
            data.create_dataset('velocity x3', data=V[vx3])
            origin.append(grid.x3min)
            extent.append(grid.x3max)
            resolution.append(grid.nx3)

        data.attrs['origin'] = origin[::-1]
        data.attrs['extent'] = extent[::-1]
        data.attrs['resolution'] = resolution[::-1]
        data.attrs['time'] = grid.t
        data.attrs['dt'] = grid.dt
        data.attrs['output_number'] = self._output_number

        data.close()
        return


    def _write_vtk(self, V, grid, algorithm, parameter):
        if parameter['Dimensions'] == '2D':
            V_vtk = np.expand_dims(V, axis=4)
            V_vtk_rho = np.copy(
                np.swapaxes(V_vtk, 1, 2)[rho, grid.jbeg:grid.jend, grid.ibeg:grid.iend, :],
                order='F')
            V_vtk_prs = np.copy(
                np.swapaxes(V_vtk, 1, 2)[prs, grid.jbeg:grid.jend, grid.ibeg:grid.iend, :],
                order='F')
            V_vtk_vx1 = np.copy(
                np.swapaxes(V_vtk, 1, 2)[vx1, grid.jbeg:grid.jend, grid.ibeg:grid.iend, :],
                order='F')
            V_vtk_vx2 = np.copy(
                np.swapaxes(V_vtk, 1, 2)[vx2, grid.jbeg:grid.jend, grid.ibeg:grid.iend, :],
                order='F')
            evtk.hl.imageToVTK(
                self._file_name,
                origin = (grid.x1[grid.ibeg], grid.x2[grid.jbeg], 0.0),
                spacing = (grid.dx1, grid.dx2, 0.0),
                cellData = {"rho":V_vtk_rho,
                            "prs":V_vtk_prs,
                            "vx1":V_vtk_vx1,
                            "vx2":V_vtk_vx2}
            )
        return

        if parameter['Dimensions'] == '3D':
            evtk.hl.gridToVTK(
                self._file_name,
                grid.x1_verts,
                grid.x2_verts,
                grid.x3_verts,
                cellData = {"rho":V[rho].T,
                            "prs":V[prs].T,
                            "vx1":V[vx1].T,
                            "vx2":V[vx2].T,
                            "vx3":V[vx3].T}
            )


    def _write_numpy(self, V, grid, algorithm, parameter):
        if parameter['Dimensions'] == '1D':
            np.save(self._file_name, (V, grid.x1))
        if parameter['Dimensions'] == '2D':
            np.save(self._file_name, (V, grid.x1, grid.x2))
        if parameter['Dimensions'] == '3D':
            np.save(self._file_name, (V, grid.x1, grid.x2, grid.x3))
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

        parameter['x1 min'] = data.attrs['origin'][-1]
        parameter['x1 max'] = data.attrs['extent'][-1]
        parameter['resolution x1'] = data.attrs['resolution'][-1]
        array_size.append(density.shape[-1])
        if parameter['Dimensions'] == "2D":
            velocity_x2 = data['velocity x2'][()]
            parameter['x2 min'] = data.attrs['origin'][-2]
            parameter['x2 max'] = data.attrs['extent'][-2]
            parameter['resolution x2'] = data.attrs['resolution'][-2]
            array_size.append(density.shape[-2])
        if parameter['Dimensions'] == "3D":
            velocity_x3 = data['velocity x3'][()]
            parameter['x3 min'] = data.attrs['origin'][-3]
            parameter['x3 max'] = data.attrs['extent'][-3]
            parameter['resolution x3'] = data.attrs['resolution'][-3]
            array_size.append(density.shape[-3])

        self._output_number = data.attrs['output_number']
        parameter['initial t'] = data.attrs['time']
        parameter['initial dt'] = data.attrs['dt']

        V = np.zeros(array_size, dtype=np.float64)

        V[rho] = density
        V[prs] = pressure
        V[vx1] = velocity_x1
        if parameter['Dimensions'] == "2D":
            V[vx2] = velocity_x2
        if parameter['Dimensions'] == "3D":
            V[vx3] = velocity_x3

        return V


    def _read_numpy(self):
        V, x1, x2, x3 = np.load()
        None
