
__author__ = "Simon Daley-Yates"
__version__ = "0.2"
__license__ = "MIT"

import numpy as np
import sys
from datetime import datetime
from mpi4py import MPI

from timer import Timer
from log import Log
from grid import Grid
from algorithms import Algorithm
from output import OutputInput
from tools import prims_to_cons


def main_loop(problem):
    """
    Synopsis
    --------
    Evolve the (M)HD equations through t = 0 to t = "max time".

    Args
    ----
    problem: object-like.
    User defined simulation problem.

    Attributes
    ----------
    None.

    TODO
    ----
    None.
    """

    #  Create MPI decomposition and get coords.
    if problem.parameter['MPI']:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        comm = None
        rank = 0

    if rank == 0:
        timing = Timer(problem.parameter)
        timing.start_sim()
        log = Log(problem.parameter)
        log.logo()
        log.options()
        io = OutputInput(problem.parameter)

    if problem.parameter['restart file'] is not None:
        V = io.input(problem.parameter)

    #  Initialise grid.
    grid = Grid(problem.parameter)
    decomp = grid.create_mpi_comm(comm, problem.parameter)

    #  Initialise Algorithms.
    algorithm = Algorithm(problem.parameter)

    #  Generate state vector to hold conservative
    #  and primative variables.
    #  Initialise the state vector accourding to
    #  user defined problem.
    if problem.parameter['restart file'] is None:
        V = grid.state_vector()
        problem.initialise(V, grid)

    #  Apply boundary conditions.
    grid.boundary(V, decomp)
    if rank == 0:
        print("")

    #  Check initial grid for nans.
    if np.isnan(np.sum(V)):
        print("Error, nan in initial conditions, exiting.")
        print("")
        sys.exit()

    U = np.empty(shape=V.shape, dtype=np.float64)
    prims_to_cons(V, U, algorithm.igamma_1)
    del V

    #  First output.
    if rank == 0:
        timing.start_io()
        io.output(U, grid, algorithm, problem.parameter)
        timing.stop_io()
        print("")

    log.begin()

    while grid.t < grid.t_max:

        if rank == 0:
            timing.start_step()
        U = algorithm.time_incriment(
            U, grid, algorithm, decomp, timing, problem.parameter
        )
        if rank == 0:
            timing.stop_step()

        log.step(grid, timing)

        if rank == 0:
            timing.start_io()
            io.output(
                U, grid, algorithm, problem.parameter
            )
        if rank == 0:
            timing.stop_io()

        grid.update_dt(decomp)

    else:

        timing.start_step()
        U = algorithm.time_incriment(
            U, grid, algorithm, decomp, timing, problem.parameter
        )
        timing.stop_step()

        log.step(grid, timing)

    timing.start_io()
    io.output(
        U, grid, algorithm, problem.parameter
    )
    timing.stop_io()

    timing.stop_sim()

    log.end(timing)

    return U





def output(A, grid, decomp):

    size = decomp.Get_size()
    rank = decomp.Get_rank()

    if grid.ndims == 2:
        sendbuf = A[: grid.gz:-grid.gz].copy()
        recv_shape = [size, sendbuf.shape[0], sendbuf.shape[1]]
    if grid.ndims == 3:
        sendbuf = A[:, grid.gz:-grid.gz, grid.gz:-grid.gz].copy()
        recv_shape = [size, sendbuf.shape[0], sendbuf.shape[1], sendbuf.shape[2]]
    if grid.ndims == 4:
        sendbuf = A[:, grid.gz:-grid.gz, grid.gz:-grid.gz, grid.gz:-grid.gz].copy()
        recv_shape = [size, sendbuf.shape[0], sendbuf.shape[1], sendbuf.shape[2], sendbuf.shape[3]]

    if rank == 0:
        recvbuf = np.empty(recv_shape, dtype='i')
    else:
        recvbuf = None

    decomp.Gather(sendbuf, recvbuf, root=0)

    global_shape = np.array(sendbuf.shape)
    global_shape *= np.append(1, grid.mpi_decomp)
    A_global = np.empty(shape=global_shape, dtype='i')

    coord_record = [decomp.Get_coords(rank_cd) for rank_cd in range(size)]

    if rank == 0:

        for i, coords in enumerate(coord_record):

            start = [coords[o]*(grid.res[o+1] - 2*grid.gz) for o in range(grid.mpi_ndims)]

            end = [start[o] + grid.res[o+1] - 2*grid.gz for o in range(grid.mpi_ndims)]

            if grid.ndims == 2:
                A_global[:, start[0]:end[0]] = recvbuf[i]

            if grid.ndims == 3:
                A_global[:, start[0]:end[0], start[1]:end[1]] = recvbuf[i]

            if grid.ndims == 4:
                A_global[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]] = recvbuf[i]

    if rank == 0:
        print('')
        print('global array:')
        print(A_global)
        print('')
        sys.stdout.flush()

    #h5py_io(A_global, decomp)

    return A_global
