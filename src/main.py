#!/usr/bin/env python3
 
__author__ = "Simon Daley-Yates"
__version__ = "0.1"
__license__ = "MIT"

import numpy as np
from grid import Grid
from user import User
from tools import time_step
from evolve import incriment
from states import States
from output import mesh_plot, line_plot, numpy_dump
import sys

def Main(u):

    print("")

    # Initialise g.
    print("    Creating g...")
    g = Grid(u.p)

    # Generate s vector to hold variables.
    print("    Creating arrays...")
    V = g.state_vector(u.p)

    # Initialise the s vector.
    print("    Initialising data...")
    u.initialise(V, g)

    if np.isnan(np.sum(V)):
        print("Error, nan in array, function: main")
        sys.exit()

    # Apply boundary conditions.
    print("    Applying boundary conditions...")
    g.boundary(V, u.p)

    # Check g.
    print("    Plotting initial conditions...")
    numpy_dump(V, g, 0)

    # Integrate in time.
    print("    Creating simulation...")
    s = States()

    print("    Starting main loop...")
    print("")

    # Perform main integration loop.
    t = 0.0
    i = 0
    num = 1

    while t < u.p['max time']:

        percent = 100.0*(t/u.p['max time'])

        if t == 0.0:
            dt = u.p['initial dt']

        dt_new, max_velocity, mach_number = time(V, g, u.p)
        dt = min(dt_new, u.p['max dt increase']*dt)

        print(f'    Iteration:{i}, t = {t:.2e}, dt = {dt:.2e}, {percent:.1f}% [{max_velocity:.5f}, {mach_number:.5f}]')

        V = incriment(V, dt, s, g, u.p)
        g.boundary(V, u.p)


        if t + dt > u.p['max time']:
            dt = u.p['max time'] - t

        if t + dt > num*u.p['plot frequency']:
            print(f"    Writing output file: {num:04}")
            numpy_dump(V, g, num)
            num += 1

        t += dt
        i += 1

    else:

        percent = 100.0*(t/u.p['max time'])
        print(f'    Iteration:{i}, t = {t:.2e}, dt = {dt:.2e}, {percent:.1f}% [{max_velocity:.5f}, {mach_number:.5f}]')

        V = incriment(V, dt, s, g, u.p)
        g.boundary(V, u.p)

        print(f"    Writing output file: {num:04}")
        numpy_dump(V, g, num)

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

    return dt, max_velocity, mach_number

