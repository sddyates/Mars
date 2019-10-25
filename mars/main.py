
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

    #sim_start_time = datetime.now().minute*60.0\
    #    + datetime.now().second\
    #    + datetime.now().microsecond*1.0e-6

    sim_start_time = datetime.now()

    logging = Log(problem.parameter)

    logging.logo()

    logging.options()

    #  Initialise grid.
    grid = Grid(problem.parameter)

    #  Initialise Algorithms.
    print("    Assigning algorithms...")
    a = Algorithm(problem.parameter)

    #  Generate state vector to hold conservative
    #  and primative variables.
    print("    Creating arrays...")
    V = grid.state_vector(problem.parameter)

    #  Initialise the state vector accourding to
    #  user defined problem.
    print("    Initialising grid...")
    problem.initialise(V, grid)

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
    percent = 100.0/problem.parameter['max time']

    #  Integrate in time.
    logging.begin()
    while t < problem.parameter['max time']:

        start_time = datetime.now().minute*60.0\
            + datetime.now().second\
            + datetime.now().microsecond*1.0e-6

        U = a.time_incriment(U, dt, grid, a, problem.parameter)

        end_time = datetime.now().minute*60.0\
            + datetime.now().second\
            + datetime.now().microsecond*1.0e-6

        time_tot = (end_time - start_time)
        Mcell = grid.rez/1.0e+6/time_tot
        Mcell_av += Mcell
        step_av += time_tot

        dt = time_step(t, grid, a, problem.parameter)

        logging.step(i, t, dt, Mcell, time_tot)

        if (problem.parameter['plot frequency'] > 0.0) &\
            ((t + dt) > num*problem.parameter['plot frequency']):
            dump(U, grid, a, problem.parameter, num)
            num += 1

        t += dt
        i += 1

    else:

        start_time = datetime.now().minute*60.0\
            + datetime.now().second\
            + datetime.now().microsecond*1.0e-6

        U = a.time_incriment(U, dt, grid, a, problem.parameter)

        end_time = datetime.now().minute*60.0\
            + datetime.now().second\
            + datetime.now().microsecond*1.0e-6

        time_tot = (end_time - start_time)
        Mcell = grid.rez/1.0e+6/time_tot
        Mcell_av += Mcell

        logging.step(i, t, dt, Mcell, time_tot)

        if problem.parameter['plot frequency'] > 0.0:
            dump(U, grid, a, problem.parameter, num)

        i+1

    #sim_end_time = datetime.now().minute*60.0\
    #    + datetime.now().second\
    #    + datetime.now().microsecond*1.0e-6

    sim_end_time = datetime.now()

    sim_time_tot = (sim_end_time - sim_start_time)

    logging.end(sim_time_tot, Mcell_av, step_av)
