
# Mars
This Code is intended to be used as a learning code and test bed for python optimisation via the numpy, numba, cython and scipy modules.

Mars is inspired by the code [PLUTO](http://plutocode.ph.unito.it/) and makes use of algorithms and from that project.

The code uses a single or two step predictor corrector method with an approximate Riemann solver, based on reconstructed fluid states of first or second order accuracy.

Please see the [Wiki](https://github.com/sddyates/mars/wiki) pages for documentation

## Todo
### Code Infrastructure
- [ ] Improve io class.
- [ ] Improve cython performance.
- [x] Restart files.
- [ ] Distributed memory parallel (MPI)
- [ ] Shared memory parallel (numba prange)
- [ ] Implement non regular grid
- [ ] User Boundary conditions
### Extra Physics
- [ ] Cooling Physics
- [ ] User source terms
- [ ] MHD
