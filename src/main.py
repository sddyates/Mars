#!/usr/bin/env python3
 
__author__ = "Simon Daley-Yates"
__version__ = "0.1"
__license__ = "MIT"

import numpy as np
from grid import Grid
from user import User
from evolve import Evolve
from output import mesh_plot, line_plot
import sys

def main():

    print("")

    # Create user.
    print("    Creating user...")
    me = User()

    # Initialise grid.
    print("    Creating grid...")
    grid = Grid(me.p)

    # Generate state vector to hold variables.
    print("    Creating arrays...")
    V = grid.state_vector(me.p)

    # Initialise the state vector.
    print("    Initialising data...")
    me.initialise(V, grid)

    if np.isnan(np.sum(V)):
        print("Error, nan in array, function: main")
        sys.exit()

    # Apply boundary conditions.
    print("    Applying boundary conditions...")
    grid.boundary(V, me.p)

    # Check grid.
    print("    Plotting initial conditions...")
    if me.p['Dimensions'] == '2D':
        #mesh_plot(V, grid, 0)
        line_plot(V[:, int(grid.nx1/2), :], grid, 0)
    if me.p['Dimensions'] == '1D':
        line_plot(V, grid, 0)

    # Integrate in time.
    print("    Creating simulation...")
    evolution = Evolve()

    print("    Starting main loop...")
    print("")

    # Perform main integration loop.
    t = 0.0
    i = 0
    num = 1

    while t < me.p['max time']:

        percent = 100.0*(t/me.p['max time'])

        if t == 0.0:
            dt = me.p['initial dt']

        dt_new, max_velocity, mach_number = time(evolution, V, grid, me.p)
        dt = min(dt_new, me.p['max dt increase']*dt)

        print(f'    Iteration:{i}, t = {t:.2e}, dt = {dt:.2e}, {percent:.1f}% [{max_velocity:.5f}, {mach_number:.5f}]')

        V = evolution.incriment(V, dt, grid, me.p)
        grid.boundary(V, me.p)


        if t + dt > me.p['max time']:
            dt = me.p['max time'] - t

        if t + dt > num*me.p['plot frequency']:
            print(f"    Writing output file: {num:04}")
            if me.p['Dimensions'] == '2D':
                #mesh_plot(V, grid, num)
                line_plot(V[:, int(grid.nx1/2), :], grid, num)
            if me.p['Dimensions'] == '1D':
                line_plot(V, grid, num)
            num += 1

        t += dt
        i += 1

    else:

        percent = 100.0*(t/me.p['max time'])
        print(f'    Iteration:{i}, t = {t:.2e}, dt = {dt:.2e}, {percent:.1f}% [{max_velocity:.5f}, {mach_number:.5f}]')

        V = evolution.incriment(V, dt, grid, me.p)
        grid.boundary(V, me.p)

        print(f"    Writing output file: {num:04}")
        if me.p['Dimensions'] == '2D':
            mesh_plot(V, grid, num)
            line_plot(V[:, int(grid.nx1/2), :], grid, num)
        if me.p['Dimensions'] == '1D':
            line_plot(V, grid, num)

    print("")
    print('    Simulation complete...')
    print("")


def time(evolution, V, grid, p):

    if p['Dimensions'] == '1D':
        dt, max_velocity, mach_number = evolution.time_step(
            V[:, grid.ibeg:grid.iend], 
            grid, 
            p['gamma'], 
            p['cfl'], 
            p['Dimensions'])
    elif p['Dimensions'] == '2D':
        dt, max_velocity, mach_number = evolution.time_step(
            V[:, grid.jbeg:grid.jend, grid.ibeg:grid.iend], 
            grid, 
            p['gamma'], 
            p['cfl'], 
            p['Dimensions'])

    return dt, max_velocity, mach_number


main()



