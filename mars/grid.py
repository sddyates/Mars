
import numpy as np
import sys
from mpi4py import MPI


class Grid:

    """
    Synopsis
    --------
    Construct variables and structures that define the grid
    and vectors for the variables and dimensions.

    Args
    ----
    p: dic-like
    Dictionary of user defined parameters, e.g.
    maximum simulation time.

    Attributes
    ----------
    state_vector()
    Define the vector structure for the variables.

    boundary()
    Assignes the boundary condtions.

    TODO
    ----
    Expand the definitions to 3D.
    """

    def __init__(self, p):

        self.ndims = np.int(p['Dimensions'][0]) + 1

        self.speed_max = np.float64(0.0)
        self.cfl = np.float64(p['cfl'])
        self.small_dt = np.float64(1.0e-12)
        self.dt = np.float64(p['initial dt'])
        self.ddt = np.float64(p['max dt increase'])
        self.t_max = np.float64(p['max time'])
        self.t = np.float64(p['initial t'])
        self.vxntb = [2, 3, 4]
        self.bc_type = np.array(p['boundaries'])

        if p['reconstruction'] == 'flat':
            self.gz = 1
        elif p['reconstruction'] == 'linear':
            self.gz = 2
        elif p['reconstruction'] == 'parabolic':
            self.gz = 3

        if self.ndims == 2:

            if p['method'] == 'hydro':
                self.nvar = 3
            elif p['method'] == 'mhd':
                self.nvar = 4

            self.x1min = p['x1 min']
            self.x1max = p['x1 max']

            self.nx1 = p['resolution x1']
            self.rez = self.nx1

            self.state_vector_shape = (
                self.nvar,
                2*self.gz + self.nx1
            )

            self.dx1 = (abs(self.x1min) + abs(self.x1max))/self.nx1
            self.dxi = [self.dx1]
            self.min_dxi = np.amin(self.dxi)

            self.ibeg = self.gz
            self.iend = self.nx1 + self.gz

            self.lower_bc_ibeg = 0
            self.lower_bc_iend = self.gz - 1

            self.upper_bc_ibeg = self.nx1 + self.gz
            self.upper_bc_iend = self.nx1 + 2*self.gz - 1

            self.imax = self.upper_bc_iend

            self.shape_internal = [self.nvar, self.nx1]
            self.shape_flux_x1 = [self.nvar, self.nx1 + 1]

            self.shape_flux = [[self.nvar, self.nx1 + 1]]

            self.x1 = self._x1()

            self.x1_verts = self._x1_verts()

        if self.ndims == 3:

            if p['method'] == 'hydro':
                self.nvar = 4
            elif p['method'] == 'mhd':
                self.nvar = 6

            self.x1min = p['x1 min']
            self.x1max = p['x1 max']
            self.x2min = p['x2 min']
            self.x2max = p['x2 max']

            self.nx1 = p['resolution x1']
            self.nx2 = p['resolution x2']
            self.rez = self.nx1*self.nx2

            self.state_vector_shape = (
                self.nvar,
                2*self.gz + self.nx2,
                2*self.gz + self.nx1
            )

            self.dx1 = (abs(self.x1min) + abs(self.x1max))/self.nx1
            self.dx2 = (abs(self.x2min) + abs(self.x2max))/self.nx2
            self.dxi = [self.dx1, self.dx2]
            self.min_dxi = np.amin(self.dxi)

            self.da = self.dx1*self.dx2

            self.ibeg = self.gz
            self.iend = self.nx1 + self.gz
            self.jbeg = self.gz
            self.jend = self.nx2 + self.gz

            self.lower_bc_ibeg = 0
            self.lower_bc_iend = self.gz - 1
            self.lower_bc_jbeg = 0
            self.lower_bc_jend = self.gz - 1

            self.upper_bc_ibeg = self.nx1 + self.gz
            self.upper_bc_iend = self.nx1 + 2*self.gz - 1
            self.upper_bc_jbeg = self.nx2 + self.gz
            self.upper_bc_jend = self.nx2 + 2*self.gz - 1

            self.jmax = self.upper_bc_jend

            self.shape_internal = [self.nvar, self.nx2, self.nx1]
            self.shape_flux_x1 = [self.nvar, self.nx1 + 1]
            self.shape_flux_x2 = [self.nvar, self.nx2 + 1]

            self.shape_flux = [[self.nvar, self.nx1 + 1],
                               [self.nvar, self.nx2 + 1]]

            self.x1 = self._x1()
            self.x2 = self._x2()

            self.x1_verts = self._x1_verts()
            self.x2_verts = self._x2_verts()

        if self.ndims == 4:

            if p['method'] == 'hydro':
                self.nvar = 5
            elif p['method'] == 'mhd':
                self.nvar = 8

            self.x1min = p['x1 min']
            self.x1max = p['x1 max']
            self.x2min = p['x2 min']
            self.x2max = p['x2 max']
            self.x3min = p['x3 min']
            self.x3max = p['x3 max']

            self.nx1 = p['resolution x1']
            self.nx2 = p['resolution x2']
            self.nx3 = p['resolution x3']
            self.rez = self.nx1*self.nx2*self.nx3

            self.state_vector_shape = (
                self.nvar,
                2*self.gz + self.nx3,
                2*self.gz + self.nx2,
                2*self.gz + self.nx1
            )

            self.dx1 = (abs(self.x1min) + abs(self.x1max))/self.nx1
            self.dx2 = (abs(self.x2min) + abs(self.x2max))/self.nx2
            self.dx3 = (abs(self.x3min) + abs(self.x3max))/self.nx3
            self.dxi = [self.dx1, self.dx2, self.dx3]
            self.min_dxi = np.amin(self.dxi)

            self.dv = self.dx1*self.dx2*self.dx3

            self.ibeg = self.gz
            self.iend = self.nx1 + self.gz
            self.jbeg = self.gz
            self.jend = self.nx2 + self.gz
            self.kbeg = self.gz
            self.kend = self.nx3 + self.gz

            self.lower_bc_ibeg = 0
            self.lower_bc_iend = self.gz - 1
            self.lower_bc_jbeg = 0
            self.lower_bc_jend = self.gz - 1
            self.lower_bc_kbeg = 0
            self.lower_bc_kend = self.gz - 1

            self.upper_bc_ibeg = self.nx1 + self.gz
            self.upper_bc_iend = self.nx1 + 2*self.gz - 1
            self.upper_bc_jbeg = self.nx2 + self.gz
            self.upper_bc_jend = self.nx2 + 2*self.gz - 1
            self.upper_bc_kbeg = self.nx3 + self.gz
            self.upper_bc_kend = self.nx3 + 2*self.gz - 1

            self.imax = self.upper_bc_iend
            self.jmax = self.upper_bc_jend
            self.kmax = self.upper_bc_kend

            self.shape_internal = [self.nvar, self.nx3, self.nx2, self.nx1]
            self.shape_flux_x1 = [self.nvar, self.nx1 + 1]
            self.shape_flux_x2 = [self.nvar, self.nx2 + 1]
            self.shape_flux_x3 = [self.nvar, self.nx3 + 1]

            self.shape_flux = [[self.nvar, self.nx1 + 1],
                               [self.nvar, self.nx2 + 1],
                               [self.nvar, self.nx3 + 1]]

            self.x1 = self._x1()
            self.x2 = self._x2()
            self.x3 = self._x3()

            self.x1_verts = self._x1_verts()
            self.x2_verts = self._x2_verts()
            self.x3_verts = self._x3_verts()

    def _x1(self):
        a = self.x1min - self.dx1*self.gz
        b = self.x1max + self.dx1*self.gz
        c = self.nx1 + 2*self.gz
        return np.linspace(a, b, c)


    def _x1_verts(self):
        a = self.x1 - self.dx1/2.0
        b = self.x1[-1] + self.dx1/2.0
        return np.append(a, b)


    def _x2(self):
        a = self.x2min - self.dx2*self.gz
        b = self.x2max + self.dx2*self.gz
        c = self.nx2 + 2*self.gz
        return np.linspace(a, b, c)


    def _x2_verts(self):
        a = self.x2 - self.dx2/2.0
        b = self.x2[-1] + self.dx2/2.0
        return np.append(a, b)


    def _x3(self):
        a = self.x3min - self.dx3*self.gz
        b = self.x3max + self.dx3*self.gz
        c = self.nx3 + 2*self.gz
        return np.linspace(a, b, c)


    def _x3_verts(self):
        a = self.x3 - self.dx3/2.0
        b = self.x3[-1] + self.dx3/2.0
        return np.append(a, b)


    def state_vector(self):
        return np.zeros(shape=self.state_vector_shape)
        # if p['Dimensions'] == '1D':
        #     return np.zeros((self.nvar,
        #                      2*self.gz + self.nx1),
        #                      dtype=np.float64)
        # elif p['Dimensions'] == '2D':
        #     return np.zeros((self.nvar,
        #                      2*self.gz + self.nx2,
        #                      2*self.gz + self.nx1),
        #                      dtype=np.float64)
        # elif p['Dimensions'] == '3D':
        #     return np.zeros((self.nvar,
        #                      2*self.gz + self.nx3,
        #                      2*self.gz + self.nx2,
        #                      2*self.gz + self.nx1),
        #                      dtype=np.float64)
        # else:
        #     print('Error, invalid number of dimensions.')
        #     sys.exit()
        # return


    def build_fluxes(self, vxn):
        if vxn == 2:
            array_shape = self.shape_flux[0]
        if vxn == 3:
            array_shape = self.shape_flux[1]
        if vxn == 4:
            array_shape = self.shape_flux[2]

        self.flux = np.zeros(shape=array_shape)
        self.FL = np.zeros(shape=array_shape)
        self.FR = np.zeros(shape=array_shape)
        self.UL = np.zeros(shape=array_shape)
        self.UR = np.zeros(shape=array_shape)
        self.VL = np.zeros(shape=array_shape)
        self.VR = np.zeros(shape=array_shape)
        self.SL = np.zeros(shape=array_shape[1])
        self.SR = np.zeros(shape=array_shape[1])
        self.pres = np.zeros(shape=array_shape[1])
        return


    def update_dt(self, decomp):

        local_dt_new = self.cfl*self.min_dxi/self.speed_max

        if decomp != None:
            dt_new = decomp.allreduce(local_dt_new, op=MPI.MIN)
        else:
            dt_new = local_dt_new

        self.dt = min(dt_new, self.ddt*self.dt)

        if (self.t + self.dt) > self.t_max:
            self.dt = self.t_max - self.t

        if self.dt < self.small_dt:
            print("dt to small, exiting.")
            print("")
            sys.exit()

        self.t += self.dt

        return


    def create_mpi_comm(self, comm, p):

        if comm == None:
            self.mpi_dims = [0, 0, 0]
            return None

        self.mpi_dims = np.array(p['mpi_decomp'])

        self.mpi_decomp = self.mpi_dims[self.mpi_dims > 1]
        if self.mpi_decomp.shape[0] == 0:
            self.mpi_decomp = [1]

        self.mpi_ndims = len(self.mpi_decomp)

        self.periods = [bc == 'periodic' for bc in self.bc_type]

        # print(self.mpi_dims, self.mpi_decomp, self.mpi_ndims, self.periods)

        decomp = comm.Create_cart(
            dims = self.mpi_decomp,
            periods = self.periods[-self.mpi_ndims:],
            reorder = True
        )

        self._modify_grid_for_mpi(p)

        # rank = decomp.Get_rank()
        # coord = decomp.Get_coords(rank)
        # print(f'rank: {rank}, coord: {coord}')

        return decomp


    def _modify_grid_for_mpi(self, p):

        if self.ndims == 2:
            self.x1max = p['x1 max']/self.mpi_dims[2]
            self.nx1 = np.int(p['resolution x1']/self.mpi_dims[2])

            self.ibeg = self.gz
            self.iend = self.nx1 + self.gz

            self.x1 = self._x1()
            self.x1_verts = self._x1_verts()

            self.state_vector_shape = (
                self.nvar,
                2*self.gz + self.nx1
            )

        if self.ndims == 3:
            self.x1max = p['x1 max']/self.mpi_dims[2]
            self.x2max = p['x2 max']/self.mpi_dims[1]

            self.nx1 = np.int(p['resolution x1']/self.mpi_dims[2])
            self.nx2 = np.int(p['resolution x2']/self.mpi_dims[1])

            self.ibeg = self.gz
            self.iend = self.nx1 + self.gz
            self.jbeg = self.gz
            self.jend = self.nx2 + self.gz

            self.x1 = self._x1()
            self.x2 = self._x2()

            self.x1_verts = self._x1_verts()
            self.x2_verts = self._x2_verts()

            self.state_vector_shape = (
                self.nvar,
                2*self.gz + self.nx2,
                2*self.gz + self.nx1
            )

        if self.ndims == 4:
            self.x1max = p['x1 max']/self.mpi_dims[2]
            self.x2max = p['x2 max']/self.mpi_dims[1]
            self.x3max = p['x3 max']/self.mpi_dims[0]

            self.nx1 = np.int(p['resolution x1']/self.mpi_dims[2])
            self.nx2 = np.int(p['resolution x2']/self.mpi_dims[1])
            self.nx3 = np.int(p['resolution x3']/self.mpi_dims[0])

            self.ibeg = self.gz
            self.iend = self.nx1 + self.gz
            self.jbeg = self.gz
            self.jend = self.nx2 + self.gz
            self.kbeg = self.gz
            self.kend = self.nx3 + self.gz

            self.x1 = self._x1()
            self.x2 = self._x2()
            self.x3 = self._x3()

            self.x1_verts = self._x1_verts()
            self.x2_verts = self._x2_verts()
            self.x3_verts = self._x3_verts()

            self.state_vector_shape = (
                self.nvar,
                2*self.gz + self.nx3,
                2*self.gz + self.nx2,
                2*self.gz + self.nx1
            )

        return


    def boundary(self, A, decomp):

        if decomp != None:
            rank = decomp.Get_rank()
            coord = np.array(decomp.Get_coords(rank))

        for dim in range(1, self.ndims):

            if self.mpi_dims[dim-1] > 1:

                self._internal_swapping(A, decomp, dim)

                if self.periods[dim-1] == False:

                    if coord[dim-1] == 0:
                        self._outflow_left(A, dim)

                    if coord[dim-1] == self.mpi_dims[dim-1] - 1:
                        self._outflow_right(A, dim)

            else:

                if self.bc_type[dim-1] == 'periodic':
                    self._periodic_left_right(A, dim)

                if self.bc_type[dim-1] == 'outflow':
                    self._outflow_left(A, dim)
                    self._outflow_right(A, dim)

        return


    def _internal_swapping(self, A, decomp, dim):

        gz = self.gz

        left, right = decomp.Shift(dim-1, 1)

        right_send = self._expand_axes([i for i in range(-2*gz, -gz)], dim)
        sendbuf_right = np.take_along_axis(A, right_send, axis=dim)
        recvbuf_right = np.empty_like(sendbuf_right)

        left_send = self._expand_axes([i for i in range(gz, 2*gz)], dim)
        sendbuf_left = np.take_along_axis(A, left_send, axis=dim)
        recvbuf_left = np.empty_like(sendbuf_left)

        decomp.Sendrecv(sendbuf_right, right, 1, recvbuf_left, left, 1)
        decomp.Sendrecv(sendbuf_left, left, 2, recvbuf_right, right, 2)

        right_indices = self._expand_axes([i for i in range(-gz, 0)], dim)
        np.put_along_axis(A, right_indices, recvbuf_right, axis=dim)

        left_indices = self._expand_axes([i for i in range(0, gz)], dim)
        np.put_along_axis(A, left_indices, recvbuf_left, axis=dim)

        return


    def _outflow_left(self, A, dim):

        gz = self.gz

        source_indices = self._expand_axes([i for i in range(gz, 2*gz)], dim)
        bc_values = np.take_along_axis(A, source_indices, axis=dim)
        bc_indices = self._expand_axes([i for i in range(0, gz)], dim)
        np.put_along_axis(A, bc_indices, np.flip(bc_values, axis=dim), axis=dim)

        return

    def _outflow_right(self, A, dim):

        gz = self.gz

        source_indices = self._expand_axes([i for i in range(-2*gz, -gz)], dim)
        bc_values = np.take_along_axis(A, source_indices, axis=dim)
        bc_indices = self._expand_axes([i for i in range(-gz, 0)], dim)
        np.put_along_axis(A, bc_indices, np.flip(bc_values, axis=dim), axis=dim)

        return


    def _periodic_left_right(self, A, dim):

        gz = self.gz

        left_indices = self._expand_axes([i for i in range(gz, 2*gz)], dim)
        right_bc_values = np.take_along_axis(A, left_indices, axis=dim)
        right_indices = self._expand_axes([i for i in range(-gz, 0)], dim)
        np.put_along_axis(A, right_indices, right_bc_values, axis=dim)

        right_indices = self._expand_axes([i for i in range(-2*gz, -gz)], dim)
        left_bc_values = np.take_along_axis(A, right_indices, axis=dim)
        left_indices = self._expand_axes([i for i in range(0, gz)], dim)
        np.put_along_axis(A, left_indices, left_bc_values, axis=dim)

        return


    def _expand_axes(self, indices, dim):

        if self.ndims == 2:
            return np.expand_dims(
                np.array(indices),
                axis=0
            )

        if self.ndims == 3:
            axes = [2, 1]
            return np.expand_dims(
                np.expand_dims(
                    np.array(indices),
                    axis=0),
                axis=axes[dim-1]
            )

        if self.ndims == 4:
            axes = [[2, 3], [2, 1], [1, 2]]
            return np.expand_dims(
                np.expand_dims(
                    np.expand_dims(
                        np.array(indices),
                        axis=0),
                    axis=axes[dim-1][0]),
                axis=axes[dim-1][1]
        )

        return indices_expanded


    # def boundary(self, V, p):
    #
    #     if p['Dimensions'] == '1D':
    #         self._lowerX1BC(V, p['lower x1 boundary'], p['Dimensions'])
    #         self._upperX1BC(V, p['upper x1 boundary'], p['Dimensions'])
    #     elif p['Dimensions'] == '2D':
    #         self._lowerX1BC(V, p['lower x1 boundary'], p['Dimensions'])
    #         self._upperX1BC(V, p['upper x1 boundary'], p['Dimensions'])
    #         self._lowerX2BC(V, p['lower x2 boundary'], p['Dimensions'])
    #         self._upperX2BC(V, p['upper x2 boundary'], p['Dimensions'])
    #     elif p['Dimensions'] == '3D':
    #         self._lowerX1BC(V, p['lower x1 boundary'], p['Dimensions'])
    #         self._upperX1BC(V, p['upper x1 boundary'], p['Dimensions'])
    #         self._lowerX2BC(V, p['lower x2 boundary'], p['Dimensions'])
    #         self._upperX2BC(V, p['upper x2 boundary'], p['Dimensions'])
    #         self._lowerX3BC(V, p['lower x3 boundary'], p['Dimensions'])
    #         self._upperX3BC(V, p['upper x3 boundary'], p['Dimensions'])
    #     else:
    #         print('Error, invalid number of dimensions.')
    #         sys.exit()
    #
    #     return
    #
    #
    # def _lowerX1BC(self, V, bc_type, dim):
    #
    #     if bc_type == 'reciprocal' and dim == '1D':
    #         V[:, :self.gz] = \
    #             V[:, self.nx1:self.nx1 + self.gz]
    #
    #     elif bc_type == 'outflow' and dim == '1D':
    #         V[:, :self.gz] = \
    #             V[:, self.gz].reshape(self.nvar, self.gz-1)
    #
    #     elif bc_type == 'reciprocal' and dim == '2D':
    #         V[:, :, :self.gz] = \
    #             V[:, :, self.nx1:self.nx1 + self.gz]
    #
    #     elif bc_type == 'outflow' and dim == '2D':
    #
    #         for o in range(self.gz):
    #             V[:, :, o] = V[:, :, self.gz]
    #
    #         #V[:, :, :self.gz] = \
    #         #    V[:, :, self.gz]#.reshape(
    #             #(self.nvar,
    #             # 2*self.gz+self.nx2,
    #             # self.gz-1))
    #
    #     elif bc_type == 'reciprocal' and dim == '3D':
    #         V[:, :, :, :self.gz] = \
    #             V[:, :, :, self.nx1:self.nx1 + self.gz]
    #
    #     elif bc_type == 'outflow' and dim == '3D':
    #         V[:, :, :, :self.gz] = \
    #             V[:, :, :, self.gz].reshape(
    #                 (self.nvar,
    #                  2*self.gz+self.nx3,
    #                  2*self.gz+self.nx2,
    #                  self.gz-1))
    #
    #     else:
    #         print('Error, invalid lower x1 boundary.')
    #         sys.exit()
    #
    #     return
    #
    #
    # def _upperX1BC(self, V, bc_type, dim):
    #
    #     if bc_type == 'reciprocal' and dim == '1D':
    #         V[:, self.upper_bc_ibeg:] = \
    #             V[:, self.gz:self.gz + 1]
    #
    #     elif bc_type == 'outflow' and dim == '1D':
    #         V[:, self.upper_bc_ibeg:] = \
    #             V[:, self.upper_bc_ibeg - 1].reshape(
    #                 (self.nvar,
    #                  self.gz-1))
    #
    #     elif bc_type == 'reciprocal' and dim == '2D':
    #         V[:, :, self.upper_bc_ibeg:] = \
    #             V[:, :, self.gz:self.gz + 1]
    #
    #     elif bc_type == 'outflow' and dim == '2D':
    #         for o in range(self.upper_bc_ibeg, self.upper_bc_iend+1):
    #             V[:, :, o] = V[:, :, self.upper_bc_ibeg - 1]
    #
    #         #V[:, :, self.upper_bc_ibeg:] = \
    #         #    V[:, :, self.upper_bc_ibeg - 1].reshape(
    #         #        (self.nvar,
    #         #         2*self.gz+self.nx2,
    #         #         self.gz - 1))
    #
    #     elif bc_type == 'reciprocal' and dim == '3D':
    #         V[:, :, :, self.upper_bc_ibeg:] = \
    #             V[:, :, :, self.gz:self.gz + 1]
    #
    #     elif bc_type == 'outflow' and dim == '3D':
    #         V[:, :, :, self.upper_bc_ibeg:] = \
    #             V[:, :, :, self.upper_bc_ibeg - 1].reshape(
    #                 (self.nvar,
    #                  2*self.gz+self.nx3,
    #                  2*self.gz+self.nx2,
    #                  self.gz - 1))
    #
    #     else:
    #         print('Error, invalid upper x1 boundary.')
    #         sys.exit()
    #
    #     return
    #
    #
    # def _lowerX2BC(self, V, bc_type, dim):
    #
    #     if bc_type == 'reciprocal' and dim == '2D':
    #         V[:, :self.gz, :] = \
    #             V[:, self.nx2:self.nx2 + self.gz, :]
    #
    #     elif bc_type == 'outflow' and dim == '2D':
    #
    #         for o in range(self.gz):
    #             V[:, o, :] = V[:, self.gz, :]
    #
    #         #V[:, :self.gz, :] = \
    #         #    V[:, self.gz, :].reshape(
    #         #        (self.nvar,
    #         #         self.gz - 1,
    #         #         self.nx1 + 2*self.gz))
    #
    #     elif bc_type == 'reciprocal' and dim == '3D':
    #         V[:, :, :self.gz, :] = \
    #             V[:, :, self.nx2:self.nx2 + self.gz, :]
    #
    #     elif bc_type == 'outflow' and dim == '3D':
    #         V[:, :, :self.gz, :] = \
    #             V[:, :, self.gz, :].reshape(
    #                 (self.nvar,
    #                  2*self.gz+self.nx3,
    #                  self.gz - 1,
    #                  2*self.gz+self.nx1))
    #
    #     else:
    #         print('Error, invalid lower x2 boundary.')
    #         sys.exit()
    #
    #     return
    #
    #
    # def _upperX2BC(self, V, bc_type, dim):
    #
    #     if bc_type == 'reciprocal' and dim == '2D':
    #         V[:, self.upper_bc_jbeg:, :] = \
    #             V[:, self.gz:self.gz + 1, :]
    #
    #     elif bc_type == 'outflow' and dim == '2D':
    #
    #         for o in range(self.upper_bc_ibeg, self.upper_bc_iend+1):
    #             V[:, o, :] = V[:, self.upper_bc_ibeg - 1, :]
    #
    #         #V[:, self.upper_bc_jbeg:, :] = \
    #         #    V[:, self.upper_bc_jbeg - 1, :].reshape(
    #         #        (self.nvar,
    #         #         self.gz - 1,
    #         #         self.nx1 + 2*self.gz))
    #
    #     elif bc_type == 'reciprocal' and dim == '3D':
    #         V[:, :, self.upper_bc_jbeg:, :] = \
    #             V[:, :, self.gz:self.gz + 1, :]
    #
    #     elif bc_type == 'outflow' and dim == '3D':
    #         V[:, :, self.upper_bc_jbeg:, :] = \
    #             V[:, :, self.upper_bc_jbeg - 1, :].reshape(
    #                 (self.nvar,
    #                  2*self.gz+self.nx3,
    #                  self.gz - 1,
    #                  2*self.gz+self.nx1))
    #
    #     else:
    #         print('Error, invalid upper x2 boundary.')
    #         sys.exit()
    #
    #     return
    #
    #
    # def _lowerX3BC(self, V, bc_type, dim):
    #
    #     if bc_type == 'reciprocal' and dim == '3D':
    #         V[:, :self.gz, :, :] = \
    #             V[:, self.nx3:self.nx3 + self.gz, :, :]
    #
    #     elif bc_type == 'outflow' and dim == '3D':
    #         V[:, :self.gz, :, :] = \
    #             V[:, self.gz, :, :].reshape(
    #                 (self.nvar,
    #                  self.gz - 1,
    #                  2*self.gz+self.nx2,
    #                  2*self.gz+self.nx1,))
    #
    #     else:
    #         print('Error, invalid upper x3 boundary.')
    #         sys.exit()
    #
    #     return
    #
    #
    # def _upperX3BC(self, V, bc_type, dim):
    #
    #     if bc_type == 'reciprocal' and dim == '3D':
    #         V[:, self.upper_bc_kbeg:, :, :] = \
    #             V[:, self.gz:self.gz + 1, :, :]
    #
    #     elif bc_type == 'outflow' and dim == '3D':
    #         V[:, self.upper_bc_kbeg:, :, :] = \
    #             V[:, self.upper_bc_kbeg - 1, :, :].reshape(
    #                 (self.nvar,
    #                  self.gz - 1,
    #                  2*self.gz+self.nx2,
    #                  2*self.gz+self.nx1,))
    #
    #     else:
    #         print('Error, invalid upper x3 boundary.')
    #         sys.exit()
    #
    #     return
