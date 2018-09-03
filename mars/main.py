#!/usr/bin/env python3
 
__author__ = "Simon Daley-Yates"
__version__ = "0.1"
__license__ = "MIT"

import numpy as np
from settings import *
from grid import Grid
from tools import time_step
from evolve import incriment
from output import dump
from settings import *
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
    print("    Setting up problem:", problem.parameter['Name'])
    print("")
    g = Grid(problem.parameter)

    # Generate state vector to hold conservative 
    # and primative variables.
    print("    Creating arrays...")
    V = g.state_vector(problem.parameter)

    # Initialise the state vector accourding to 
    # user defined problem.
    print("    Initialising grid...")
    problem.initialise(V, g)

    # Apply boundary conditions.
    print("    Applying boundary conditions...")
    g.boundary(V, problem.parameter)

    # Check initial grid for nans.
    if np.isnan(np.sum(V)):
        print("Error, nan in array, function: main")
        sys.exit()

    # Integrate in time.
    print("    Starting time integration loop...")
    print("")
    
    # First output.
    dump(V, g, problem.parameter, 0)

    # Perform main integration loop.
    t = 0.0
    dt = problem.parameter['initial dt']
    i = 0
    num = 1
    percent = 100.0/problem.parameter['max time']

    while t < problem.parameter['max time']:

        V = incriment(V, dt, g, problem.parameter)

        dt_new, max_velocity, mach_number = time_step(V, g, problem.parameter)

        print(f"    n = {i}, t = {t:.2e}, dt = {dt:.2e}, "
        + f"{percent*t:.1f}% [{max_velocity:.1f}, {mach_number:.1f}]")

        dt = min(dt_new, problem.parameter['max dt increase']*dt)

        if (t + dt) > problem.parameter['max time']:
            dt = problem.parameter['max time'] - t

        if (t + dt) > num*problem.parameter['plot frequency']:
            dump(V, g, problem.parameter, num)
            num += 1

        t += dt
        i += 1

    else:
        print(f"    n = {i}, t = {t:.2e}, dt = {dt:.2e}, "
        + f"{percent*t:.1f}% [{max_velocity:.1f}, {mach_number:.1f}]")

        V = incriment(V, dt, g, problem.parameter)
        dump(V, g, problem.parameter, num)

    print("")
    print('    Simulation complete...')
    print("")