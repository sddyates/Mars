__author__ = "Simon Daley-Yates"
__version__ = "1.0"
__license__ = "MIT"

import sys
import numpy as np
#  from datetime import datetime
from mpi4py import MPI

from .timer import Timer
from .log import Log
from .grid import Grid
from .algorithms import Algorithm
from .output import OutputInput
from .tools import prims_to_cons


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
    # if problem.parameter['MPI']:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # else:
    # comm = None
    # rank = 0

    if rank == 0:
        timing = Timer(problem.parameter)
        timing.start_sim()
        log = Log(problem.parameter)
        log.logo()
        log.options()
    else:
        timing = None

    io = OutputInput(problem.parameter)

    if problem.parameter['restart file'] is not None:
        V = io.input(problem.parameter)

    #  Initialise grid.
    grid = Grid(problem.parameter, comm)

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
    if rank == 0:
        timing.start_boundary()
    grid.boundary(V)
    if rank == 0:
        timing.stop_boundary()

    if rank == 0:
        print("")
        sys.stdout.flush()

    #  Check initial grid for nans.
    if np.isnan(np.sum(V)):
        print("Error, nan in initial conditions, exiting.")
        print("")
        sys.stdout.flush()
        sys.exit()

    U = np.empty(shape=V.shape, dtype=np.float64)
    prims_to_cons(V, U, algorithm.igamma_1)
    del V

    #  First output.
    if rank == 0:
        timing.start_io()
    io.output(U, grid, algorithm, problem.parameter)
    if rank == 0:
        timing.stop_io()

    if rank == 0:
        print("")
        sys.stdout.flush()
        log.begin()

    while grid.t < grid.t_max:

        if rank == 0:
            timing.start_step()
        U = algorithm.time_incriment(
            U, grid, algorithm, timing, problem.parameter
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

        grid.update_dt()

    else:

        if rank == 0:
            timing.start_step()
        U = algorithm.time_incriment(
            U, grid, algorithm, timing, problem.parameter
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
        timing.stop_sim()
        log.end(timing)

    return U
