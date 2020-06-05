## Todo
### Code Infrastructure
- [ ] Improve io class.
- [x] Restart files.
- [ ] Distributed memory parallel (MPI)
- [ ] Shared memory parallel (numba prange)
- [ ] Implement non regular grid
- [ ] User Boundary conditions
### Extra Physics
- [ ] Cooling Physics
- [ ] User source terms
- [ ] MHD

## For MPI implementation

### Boundaries
- [x] internal boundaries
- [x] external periodic
- [x] external outflow
- [ ] external reflective

### IO
Need to add method to io class that reduces separate ranks before sending global grid to the serial output routines.

- [ ] reduction before IO
- [ ] parallel IO

### Grid modifications
Need to add method to modify serial grid variables as a function of the MPI decomposition.

- [ ] Second state vector method which creates the local grids.
- [ ] modify the loop limits for the spatial loops.

### Time step calculation
- [x] Add version to all reduce new_dt and do a min.
