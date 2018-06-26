#!/usr/bin/env python3
 
__author__ = "Simon Daley-Yates"
__version__ = "0.1"
__license__ = "MIT"

import numpy as np
from settings import *
from grid import Grid
from tools import time_step
from evolve import incriment
from output import numpy_dump
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

    # Initialise g.
    print("    Creating grid...")
    g = Grid(problem.parameter)

    # Generate s vector to hold variables.
    print("    Creating arrays...")
    V = g.state_vector(problem.parameter)

    # Initialise the state vector.
    print("    Initialising data...")
    problem.initialise(V, g)

    if np.isnan(np.sum(V)):
        print("Error, nan in array, function: main")
        sys.exit()

    # Apply boundary conditions.
    print("    Applying boundary conditions...")
    g.boundary(V, problem.parameter)

    # Check g.
    print("    Plotting initial conditions...")
    numpy_dump(V, g, problem.parameter, 0)

    # Integrate in time.
    print("    Starting main loop...")
    print("")

    # Perform main integration loop.
    t = 0.0
    i = 0
    num = 1

    while t < problem.parameter['max time']:

        percent = 100.0*(t/problem.parameter['max time'])

        if t == 0.0:
            dt = problem.parameter['initial dt']

        dt_new, max_velocity, mach_number = time_step(V, g, problem.parameter)
        dt = min(dt_new, problem.parameter['max dt increase']*dt)

        print(f'    n = {i}, t = {t:.2e}, dt = {dt:.2e}, {percent:.1f}% [{max_velocity:.1f}, {mach_number:.1f}]')

        V = incriment(V, dt, g, problem.parameter)

        if t + dt > problem.parameter['max time']:
            dt = problem.parameter['max time'] - t

        if t + dt > num*problem.parameter['plot frequency']:
            print(f"    Writing output file: {num:04}")
            numpy_dump(V, g, problem.parameter, num)
            num += 1

        t += dt
        i += 1

    else:

        percent = 100.0*(t/problem.parameter['max time'])
        print(f'    n = {i}, t = {t:.2e}, dt = {dt:.2e}, {percent:.1f}% [{max_velocity:.1f}, {mach_number:.1f}]')

        V = incriment(V, dt, g, problem.parameter)

        print(f"    Writing output file: {num:04}")
        numpy_dump(V, g, problem.parameter, num)

    print("")
    print('    Simulation complete...')
    print("")