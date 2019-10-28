
# Mars
This Code is intended to be used as a learning code and test bed for python optimisation via the numpy, numba, cython and scipy modules.

Mars is inspired by the code [PLUTO](http://plutocode.ph.unito.it/) and makes use of algorithms and from that project.

## Dependencies
* Python 3.5 or greater
* Numpy
* Numba
* vtk
* h5py
* cython (optional)

Using [Anaconda python](https://www.anaconda.com/distribution/) these dependencies can be installed using the cmd

    $ conda create --name mars  # This creates an conda env for mars to run in
    $ conda activate mars
    $ conda install -c anaconda numpy numba h5py cython # installs dependencies
    $ conda install -c e3sm evtk

## Running a problem script

Running a simulation with Mars is centered around executing a problem script, i.e.

    $ python problem.py



## Todo
### Code Infrastructure
- [ ] Improve io class.
- [ ] Improve cython performance.
- [ ] Restart files.
- [ ] MPI
- [ ] Implement non regular grid
### Extra Physics
- [ ] Self-gravity.
- [ ] MHD
- [ ] AMR
