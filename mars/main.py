
__author__ = "Simon Daley-Yates"
__version__ = "0.2"
__license__ = "MIT"

import numpy as np
from settings import *
from grid import Grid
from algorithms import Algorithm
from tools import time_step, prims_to_cons
from datetime import datetime
from output import dump
from log import Log
from timer import Timer
import sys


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

    log = Log(problem.parameter)

    log.logo()

    log.options()

    #  Initialise grid.
    grid = Grid(problem.parameter, log)

    #  Initialise Algorithms.
    print("    Assigning algorithms...")
    a = Algorithm(problem.parameter, log)

    #  Generate state vector to hold conservative
    #  and primative variables.
    print("    Creating arrays...")
    V = grid.state_vector(problem.parameter)

    #  Initialise the state vector accourding to
    #  user defined problem.
    print("    Initialising grid...")
    problem.initialise(V, grid, log)

    #  Apply boundary conditions.
    print("    Applying boundary conditions...")
    grid.boundary(V, problem.parameter)
    print("")

    #  Check initial grid for nans.
    if np.isnan(np.sum(V)):
        print("Error, nan in array, function: main")
        sys.exit()

    #  First output.
    if problem.parameter['plot frequency'] > 0.0:
        dump(V, grid, a, problem.parameter, 0)
    print("")

    U = np.empty(shape=V.shape, dtype=np.float64)
    prims_to_cons(V, U, a.igamma_1)
    del V

    #  Perform main integration loop.
    t = 0.0
    dt = problem.parameter['initial dt']
    i = 0
    num = 1
    Mcell_av = 0.0
    step_av = 0.0

    #  Integrate in time.
    log.begin()
    while t < problem.parameter['max time']:

        timing.start_step
        U = a.time_incriment(U, dt, grid, a, timing, problem.parameter)
        timing.stop_step

        dt = time_step(t, grid, a, problem.parameter)

        log.step(i, t, dt, timing)

        if (problem.parameter['plot frequency'] > 0.0) &\
            ((t + dt) > num*problem.parameter['plot frequency']):
            dump(U, grid, a, problem.parameter, num)
            num += 1

        t += dt
        i += 1

    else:

        timing.start_step
        U = a.time_incriment(U, dt, grid, a, timing, problem.parameter)
        timing.stop_step

        log.step(i, t, dt, timing)

        if problem.parameter['plot frequency'] > 0.0:
            dump(U, grid, a, problem.parameter, num)

        i+1

    log.end(i, sim_time_tot, Mcell_av, step_av)
