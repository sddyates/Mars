
# Mars

This Code is intended to be used by as a learning code and test bed for python optimisation via the numpy, numba and scipy modules.

Mars is inspired by the code [PLUTO](http://plutocode.ph.unito.it/) and makes use of algorithms and code from that project.


# Todo

## Code Infrastructure

Implement the non regular grid:
    - this requires rewriting the Grid object so that dx, dy, dz are arrays
    and that vertices are arrays, this will also mean rewriting the output function.

To Do list for project:

- choose type of output via user parameters.

- make io class.

- do comparison to cython.

- Restart files.


Extra Physics
-------------
- [ ] Self-gravity.
- [ ] MHD
- [ ] MPI
- [ ] AMR
