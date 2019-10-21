#!/usr/bin/env python3

__author__ = "Simon Daley-Yates"
__version__ = "0.1"
__license__ = "MIT"

import numpy as np
from settings import *
from grid import Grid
from algorithms import Algorithm
from tools import time_step, prims_to_cons
from datetime import datetime
from output import dump
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

    # Set global parameters.
    print("")

    print(r"    -----------------------------------------------")
    print(r"                                                   ")
    print(r"        \\\\\\\\\      /\     |\\\\\ /\\\\\        ")
    print(r"        ||  ||  \\    //\\    ||  // ||            ")
    print(r"        ||  ||  ||   //  \\   ||\\\\ \\\\\\        ")
    print(r"        ||  ||  ||  //\\\\\\  ||  ||     ||        ")
    print(r"        ||  ||  || //      \\ ||  || \\\\\/ 0.2    ")
    print(r"                                                   ")
    print(r"    -----------------------------------------------")

    print(f"    Problem settings:")
    print(f"        - Name: {problem.parameter['Name']}")
    print(f"        - Max time: {problem.parameter['max time']}")
    print("")

    print("    Assigning algorithms...")
    print("    Creating arrays...")
    print("    Initialising grid...")
    print("    Applying boundary conditions...")
    print("    Starting time integration loop...")
    print("")

    # Initialise grid.
    grid = Grid(problem.parameter)

    # Initialise Algorithms.
    a = Algorithm(problem.parameter)

    # Generate state vector to hold conservative
    # and primative variables.
    V = grid.state_vector(problem.parameter)

    # Initialise the state vector accourding to
    # user defined problem.
    problem.initialise(V, grid)

    # Apply boundary conditions.
    grid.boundary(V, problem.parameter)

    # Check initial grid for nans.
    if np.isnan(np.sum(V)):
        print("Error, nan in array, function: main")
        sys.exit()

    # First output.
    if problem.parameter['plot frequency'] > 0.0:
        dump(V, grid, a, problem.parameter, 0)

    U = np.zeros(shape=V.shape, dtype=np.float64)
    prims_to_cons(V, U, a.gamma_1)
    del V

    # Perform main integration loop.
    t = 0.0
    dt = problem.parameter['initial dt']
    i = 0
    num = 1
    percent = 100.0/problem.parameter['max time']

    # Integrate in time.
    while t < problem.parameter['max time']:

        start_time = datetime.now().second + datetime.now().microsecond*1.0e-6

        U = a.time_incriment(U, dt, grid, a, problem.parameter)

        end_time = datetime.now().second + datetime.now().microsecond*1.0e-6
        time_tot = (end_time - start_time)
        Mcell = grid.rez/1.0e+6/time_tot

        dt_new, max_velocity, mach_number = time_step(grid, a)

        print(f"    n = {i}, t = {t:.2e}, dt = {dt:.2e}, "
              + f"{percent*t:.1f}% [{max_velocity:.1f}, {mach_number:.1f}], {Mcell:.3f} Mcell/s")

        dt = min(dt_new, problem.parameter['max dt increase']*dt)

        if (t + dt) > problem.parameter['max time']:
            dt = problem.parameter['max time'] - t

        if (problem.parameter['plot frequency'] > 0.0) & ((t + dt) > num*problem.parameter['plot frequency']):
            dump(U, grid, a, problem.parameter, num)
            num += 1

        t += dt
        i += 1

    else:
        print(f"    n = {i}, t = {t:.2e}, dt = {dt:.2e}, "
              + f"{percent*t:.1f}% [{max_velocity:.1f}, {mach_number:.1f}], {Mcell:.3f} Mcell/s")

        U = a.time_incriment(U, dt, grid, a, problem.parameter)

        if problem.parameter['plot frequency'] > 0.0:
            dump(U, grid, a, problem.parameter, num)

    print("")
    print(f"    Simulation {problem.parameter['Name']} complete...")
    print("")
