
__author__ = "Simon Daley-Yates"
__version__ = "0.2"
__license__ = "MIT"

import numpy as np
import sys
from datetime import datetime

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

    timing = Timer(problem.parameter)

    timing.start_simulation()

    log = Log(problem.parameter)

    log.logo(problem.parameter)

    log.options(problem.parameter)

    print("    Initialising IO...")
    io = OutputInput(problem.parameter)

    if problem.parameter['restart file'] is not None:
        V = io.input(problem.parameter)

    #  Initialise grid.
    grid = Grid(problem.parameter)

    #  Initialise Algorithms.
    print("    Assigning algorithms...")
    algorithm = Algorithm(problem)

    #  Generate state vector to hold conservative
    #  and primative variables.
    #  Initialise the state vector accourding to
    #  user defined problem.
    if problem.parameter['restart file'] is None:
        print("    Creating arrays...")
        V = grid.state_vector(problem.parameter)
        print("    Setting intial conditions...")
        problem.initialise(V, grid)

    if problem.parameter['normalise']:
        algorithm.normalise_state_variables(V)

    #  Apply boundary conditions.
    print("    Applying boundary conditions...")
    grid.boundary(V, problem.parameter)
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
    if problem.parameter['restart file'] is None:
        timing.start_io()
        io.output(U, grid, algorithm)
        timing.stop_io()
        print("")

    log.begin()

    while grid.t < grid.t_max:

        timing.start_step()
        U = algorithm.time_incriment(
            U, grid, algorithm, timing, problem.parameter
        )
        timing.stop_step()

        log.step(grid, timing, problem.parameter)

        timing.start_io()
        io.output(U, grid, algorithm)
        timing.stop_io()

        grid.update_dt()

    else:

        timing.start_step()
        U = algorithm.time_incriment(
            U, grid, algorithm, timing, problem.parameter
        )
        timing.stop_step()

        log.step(grid, timing, problem.parameter)

    timing.start_io()
    io.output(U, grid, algorithm)
    timing.stop_io()

    timing.stop_simulation()

    log.end(timing, problem.parameter)

    return U
