
import os
import sys
import numpy as np
import pyevtk
import h5py

from .settings import rho, prs, vx1, vx2, vx3
from .tools import cons_to_prims


class OutputInput:

    def __init__(self, p):

        self._output_number = 0
        self._save_interval = np.float64(p['save interval'])
        self._output_prims = p['output primitives']
        self._io_type = p['output type']
        self._io_folder = "./output/" + p['Dimensions'] + "/"

        if not os.path.isdir(self._io_folder):
            os.makedirs(self._io_folder)
            print(f'    Created io folder at: "{self._io_folder}"')

        self._file_prefix = "data_" + p['Dimensions'] + "_"

        self._base_file_name = self._io_folder + self._file_prefix

        if p['restart file'] is not None:
            self._restart_file = self._io_folder \
                + self._file_prefix \
                + f"{p['restart file']:04}" \
                + ".h5"

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

        if (self._save_interval > 0.0) & (g.t >= self._output_number*self._save_interval):

            if g.rank == 0:
                self._file_name = self._base_file_name + f"{self._output_number:04}"
                print(f"    Writing {self._io_type}: {self._output_number:04}")
                sys.stdout.flush()

            if any(g.mpi_decomposition > 1):
                U_global = self._reduce_from_mpi(U, g)
            else:
                U_global = U

            if g.rank == 0:
                if self._output_prims:
                    V = self._convert(U_global, a.gamma_1)

                for write in self.write_file:
                    write(V, g, a, p)

            self._output_number += 1

        return

    def _convert(self, U, gamma_1):
        V = np.zeros(shape=U.shape, dtype=np.float64)
        cons_to_prims(U, V, gamma_1)
        return V

    def _write_h5(self, V, g, a, p):

        data = h5py.File(self._file_name + ".h5", "w")

        data.create_dataset('density', data=V[rho])
        data.create_dataset('pressure', data=V[prs])
        data.create_dataset('velocity x1', data=V[vx1])
        origin = [g.min[0]]
        extent = [g.max[0]]
        resolution = [g.nx[0]]

        if p['Dimensions'] == "2D":
            data.create_dataset('velocity x2', data=V[vx2])
            origin.append(g.min[1])
            extent.append(g.max[1])
            resolution.append(g.nx[1])

        if p['Dimensions'] == "3D":
            data.create_dataset('velocity x3', data=V[vx3])
            origin.append(g.min[2])
            extent.append(g.max[2])
            resolution.append(g.nx[2])

        data.attrs['origin'] = origin[::-1]
        data.attrs['extent'] = extent[::-1]
        data.attrs['resolution'] = resolution[::-1]
        data.attrs['time'] = g.t
        data.attrs['dt'] = g.dt
        data.attrs['output_number'] = self._output_number

        data.close()
        return

    # def _write_vtk(self, V, g, a, p):
    #     if p['Dimensions'] == '2D':
    #         V_vtk = np.expand_dims(V, axis=3)
    #         V_vtk_rho = np.copy(
    #             np.swapaxes(V_vtk, 1, 2)[rho, g.jbeg:g.jend, g.ibeg:g.iend, :],
    #             order='F')
    #         V_vtk_prs = np.copy(
    #             np.swapaxes(V_vtk, 1, 2)[prs, g.jbeg:g.jend, g.ibeg:g.iend, :],
    #             order='F')
    #         V_vtk_vx1 = np.copy(
    #             np.swapaxes(V_vtk, 1, 2)[vx1, g.jbeg:g.jend, g.ibeg:g.iend, :],
    #             order='F')
    #         V_vtk_vx2 = np.copy(
    #             np.swapaxes(V_vtk, 1, 2)[vx2, g.jbeg:g.jend, g.ibeg:g.iend, :],
    #             order='F')
    #         pyevtk.hl.imageToVTK(
    #             self._file_name,
    #             origin = (g.x1[g.ibeg], g.x2[g.jbeg], 0.0),
    #             spacing = (g.dx1, g.dx2, 0.0),
    #             cellData = {"rho":V_vtk_rho,
    #                         "prs":V_vtk_prs,
    #                         "vx1":V_vtk_vx1,
    #                         "vx2":V_vtk_vx2}
    #         )

    def _write_vtk(self, V, g, a, p):
        if p['Dimensions'] == '2D':
            V_vtk = np.expand_dims(V, axis=3)
            V_vtk_rho = np.copy(
                np.swapaxes(V_vtk, 1, 2)[rho, :, :, :],
                order='F')
            V_vtk_prs = np.copy(
                np.swapaxes(V_vtk, 1, 2)[prs, :, :, :],
                order='F')
            V_vtk_vx1 = np.copy(
                np.swapaxes(V_vtk, 1, 2)[vx1, :, :, :],
                order='F')
            V_vtk_vx2 = np.copy(
                np.swapaxes(V_vtk, 1, 2)[vx2, :, :, :],
                order='F')
            pyevtk.hl.imageToVTK(
                self._file_name,
                origin=(g.x[1][g.beg[1]], g.x[0][g.beg[0]], 0.0),
                spacing=(g.dx[1], g.dx[0], 0.0),
                cellData={"rho": V_vtk_rho,
                          "prs": V_vtk_prs,
                          "vx1": V_vtk_vx1,
                          "vx2": V_vtk_vx2}
            )

        if p['Dimensions'] == '3D':
            pyevtk.hl.gridToVTK(
                self._file_name,
                np.array(g.x_verts_global[2][g.gz:-g.gz], dtype=np.float64),
                np.array(g.x_verts_global[1][g.gz:-g.gz], dtype=np.float64),
                np.array(g.x_verts_global[0][g.gz:-g.gz], dtype=np.float64),
                cellData={"rho": V[rho].T,
                          "prs": V[prs].T,
                          "vx1": V[vx1].T,
                          "vx2": V[vx2].T,
                          "vx3": V[vx3].T}
            )

        return

    def _write_numpy(self, V, g, a, p):
        np.save(self._file_name, (V, g.x))
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

    def _reduce_from_mpi(self, A, grid):

        size = grid.decomp.Get_size()
        rank = grid.decomp.Get_rank()

        if grid.ndims == 2:
            sendbuf = A[:, grid.gz:-grid.gz].copy()
            recv_shape = [size, sendbuf.shape[0], sendbuf.shape[1]]
        if grid.ndims == 3:
            sendbuf = A[:, grid.gz:-grid.gz, grid.gz:-grid.gz].copy()
            recv_shape = [size, sendbuf.shape[0], sendbuf.shape[1], sendbuf.shape[2]]
        if grid.ndims == 4:
            sendbuf = A[:, grid.gz:-grid.gz, grid.gz:-grid.gz, grid.gz:-grid.gz].copy()
            recv_shape = [size, sendbuf.shape[0], sendbuf.shape[1], sendbuf.shape[2], sendbuf.shape[3]]

        if rank == 0:
            recvbuf = np.empty(recv_shape, dtype=np.float64)
        else:
            recvbuf = None

        grid.decomp.Gather(sendbuf, recvbuf, root=0)

        if rank == 0:

            global_shape = np.array(sendbuf.shape)
            global_shape *= np.append(1, grid.mpi_decomposition)
            A_global = np.empty(shape=global_shape, dtype=np.float64)

            for i, coords in enumerate(grid.coord_record):

                start = [coords[o]*grid.nx[o] for o in range(len(grid.mpi_decomposition))]
                end = [start[o] + grid.nx[o] for o in range(len(grid.mpi_decomposition))]

                if grid.ndims == 2:
                    A_global[:, start[0]:end[0]] = recvbuf[i]

                if grid.ndims == 3:
                    A_global[:, start[0]:end[0], start[1]:end[1]] = recvbuf[i]

                if grid.ndims == 4:
                    A_global[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]] = recvbuf[i]

        else:
            A_global = None

        return A_global
