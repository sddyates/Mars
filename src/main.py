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
    U = grid.state_vector(me.p)

    # Initialise the state vector.
    print("    Initialising data...")
    me.initialise(U, grid)

    if np.isnan(np.sum(U)):
        print("Error, nan in array, function: main")
        sys.exit()

    # Apply boundary conditions.
    print("    Applying boundary conditions...")
    grid.boundary(U, me.p)

    # Check grid.
    print("    Plotting initial conditions...")
    #mesh_plot(U, grid, 0)
    line_plot(U, grid, 0)

    # Integrate in time.
    print("    Creating simulation...")
    evolution = Evolve()

    print("    Starting main loop...")
    print("")

    # Perform main integration loop.
    t = 0.0
    i = 0
    num = 1
    dt_new, max_velocity, mach_number = time(evolution, U, grid, me.p)
    dt = me.p['initial dt']

    while t < me.p['max time']:

        print(f'    Iteration:{i}, t = {t:.5f}, dt = {dt:.5f}, [{max_velocity:.5f}, {mach_number:.5f}]')

        U = evolution.incriment(U, dt, grid, me.p)
        grid.boundary(U, me.p)

        dt_new, max_velocity, mach_number = time(evolution, U, grid, me.p)
        dt = min(dt_new, me.p['max dt increase']*dt)

        if t + dt > me.p['max time']:
            dt = me.p['max time'] - t

        if t + dt > num*me.p['plot frequency']:
            print(f"    Writing output file: {num:04}")
            #mesh_plot(U, grid, num)
            line_plot(U, grid, num)
            num += 1

        t += dt
        i += 1

    else:

        print(f'    Iteration:{i}, t = {t:.5f}, dt = {dt:.5f}, [{max_velocity:.5f}, {mach_number:.5f}]')

        U = evolution.incriment(U, dt, grid, me.p)
        grid.boundary(U, me.p)

        print(f"    Writing output file: {num:04}")
        #Mesh_plot(U, grid, num) 
        line_plot(U, grid, num)

    print("")
    print('    Simulation complete...')
    print("")


def time(evolution, U, grid, p):

    if p['Dimensions'] == '1D':
        dt, max_velocity, mach_number = evolution.time_step(
            U[:, grid.ibeg:grid.iend], 
            grid, 
            p['gamma'], 
            p['cfl'], 
            p['Dimensions'])
    elif p['Dimensions'] == '2D':
        dt, max_velocity, mach_number = evolution.time_step(
            U[:, grid.jbeg:grid.jend, grid.ibeg:grid.iend], 
            grid, 
            p['gamma'], 
            p['cfl'], 
            p['Dimensions'])

    return dt, max_velocity, mach_number


main()



