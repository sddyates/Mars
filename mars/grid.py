
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
    """

    def __init__(self, p, comm):

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

        resolution = np.array(p['resolution'], dtype=np.int)
        mask = resolution > 1

        self.mpi_decomposition = np.array(p['mpi decomposition'], dtype=np.int)[mask]
        self.periods = np.array([bc == 'periodic' for bc in self.bc_type])[mask]

        self.decomp = comm.Create_cart(
            dims = self.mpi_decomposition,
            periods = self.periods,
            reorder = True
        )

        self.rank = self.decomp.Get_rank()
        self.comm_size = self.decomp.Get_size()
        self.mpi_coords = np.array(self.decomp.Get_coords(self.rank), dtype=np.int)

        self.nx = np.array(resolution[mask]/self.mpi_decomposition, dtype=np.int)

        max = np.array(p['max'], dtype=np.float64)[mask]
        min = np.array(p['min'], dtype=np.float64)[mask]

        extent = max - min
        self.max = min + extent/self.mpi_decomposition*(self.mpi_coords + 1)
        self.min = min + extent/self.mpi_decomposition*self.mpi_coords

        self.dx = (self.max - self.min)/self.nx

        self.beg = np.zeros_like(self.nx) + self.gz
        self.end = self.nx + self.gz

        self.x, self.x_verts = self._x()

        self.nvar = 2 + len(self.nx[self.nx > 1])

        self.min_dx = np.amin(self.dx)
        self.rez = np.prod(self.nx)

        self.coord_record = [np.array(self.decomp.Get_coords(rank_cd)) for rank_cd in range(self.comm_size)]

        self.state_vector_shape = (np.append(self.nvar, self.nx + 2*self.gz))


    def _x(self):

        a = self.min - self.dx*self.gz
        b = self.max + self.dx*self.gz
        c = self.nx + 2*self.gz

        x = []
        x_vert = []
        for i, (A, B, C) in enumerate(zip(a, b, c)):

            xi = np.linspace(A, B, C)
            x.append(xi)

            d = xi - self.dx[i]/2.0
            e = xi[-1] + self.dx[i]/2.0
            x_vert.append(np.append(d, e))

        return np.array(x), np.array(x_vert)


    def state_vector(self):
        return np.zeros(shape=self.state_vector_shape)


    def update_dt(self):

        local_dt_new = self.cfl*self.min_dx/self.speed_max

        if self.decomp != None:
            dt_new = np.array([0.0])
            self.decomp.Allreduce(local_dt_new, dt_new, MPI.MIN)
        else:
            dt_new = np.array([local_dt_new])

        self.dt = min(dt_new[0], self.ddt*self.dt)

        if (self.t + self.dt) > self.t_max:
            self.dt = self.t_max - self.t

        if self.dt < self.small_dt:
            print("dt to small, exiting.")
            print("")
            sys.exit()

        self.t += self.dt

        return


    def boundary(self, A):

        for dim in range(1, self.ndims):

            if self.mpi_decomposition[dim-1] > 1:

                self._internal_swapping(A, dim)

                if self.periods[dim-1] == False:

                    if self.mpi_coords[dim-1] == 0:
                        self._outflow_left(A, dim)

                    if self.mpi_coords[dim-1] == self.mpi_decomposition[dim-1] - 1:
                        self._outflow_right(A, dim)

            else:

                if self.bc_type[dim-1] == 'periodic':
                    self._periodic_left_right(A, dim)

                if self.bc_type[dim-1] == 'outflow':
                    self._outflow_left(A, dim)
                    self._outflow_right(A, dim)

        return


    def _internal_swapping(self, A, dim):

        gz = self.gz

        left, right = self.decomp.Shift(dim-1, 1)

        right_send = self._expand_axes([i for i in range(-2*gz, -gz)], dim)
        sendbuf_right = np.take_along_axis(A, right_send, axis=dim)
        recvbuf_right = np.empty_like(sendbuf_right)

        left_send = self._expand_axes([i for i in range(gz, 2*gz)], dim)
        sendbuf_left = np.take_along_axis(A, left_send, axis=dim)
        recvbuf_left = np.empty_like(sendbuf_left)

        self.decomp.Sendrecv(sendbuf_right, right, 1, recvbuf_left, left, 1)
        self.decomp.Sendrecv(sendbuf_left, left, 2, recvbuf_right, right, 2)

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
