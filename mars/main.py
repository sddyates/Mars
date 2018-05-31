#!/usr/bin/env python3
 
__author__ = "Simon Daley-Yates"
__version__ = "0.1"
__license__ = "MIT"

import numpy as np
from grid import Grid
from tools import time_step
from evolve import incriment
from states import States
from output import numpy_dump
import sys

def main_loop(problem):

    # Set global parameters.    
    set_globals()

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
    print("    Creating simulation...")
    s = States()

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

        dt_new, max_velocity, mach_number = time(V, g, problem.parameter)
        dt = min(dt_new, problem.parameter['max dt increase']*dt)

        print(f'    Iteration:{i}, t = {t:.2e}, dt = {dt:.2e}, {percent:.1f}% [{max_velocity:.5f}, {mach_number:.5f}]')

        V = incriment(V, dt, s, g, problem.parameter)

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
        print(f'    Iteration:{i}, t = {t:.2e}, dt = {dt:.2e}, {percent:.1f}% [{max_velocity:.5f}, {mach_number:.5f}]')

        V = incriment(V, dt, s, g, problem.parameter)

        print(f"    Writing output file: {num:04}")
        numpy_dump(V, g, problem.parameter, num)

    print("")
    print('    Simulation complete...')
    print("")


def time(V, g, p):

    if p['Dimensions'] == '1D':
        dt, max_velocity, mach_number = time_step(
            V[:, g.ibeg:g.iend], g, p)
    elif p['Dimensions'] == '2D':
        dt, max_velocity, mach_number = time_step(
            V[:, g.jbeg:g.jend, g.ibeg:g.iend], g, p)
    elif p['Dimensions'] == '3D':
        dt, max_velocity, mach_number = time_step(
            V[:, g.kbeg:g.kend, g.jbeg:g.jend, g.ibeg:g.iend], g, p)

    return dt, max_velocity, mach_number


def set_globals():
    """
    Synopsis
    --------
    Define global variables for element reference.

    Args
    ----
    None.

    Attributes
    ----------
    None.

    TODO
    ----
    None.
    """

    global rho
    global prs
    global vx1
    global vx2
    global vx3

    global eng
    global mvx1
    global mvx2
    global mvx3
    global u
    global v
    global w

    global SmallPressure

    rho = 0
    prs = 1
    vx1 = 2
    vx2 = 3
    vx3 = 4

    eng = prs
    u = mvx1 = vx1 
    v = mvx2 = vx2
    w = mvx3 = vx3

    SmallPressure = 1.0e-12