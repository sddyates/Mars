import numpy as np
from cython_lib.solvers import tvdlf, hll, hllc
from cython_lib.piecewise import flat, minmod

class Geometry:

    def __init__(self, p):
        self.direction_count = 0

        self.riemann = {'tvdlf':tvdlf,
                        'hll':hll,
                        'hllc':hllc}

        self.reconstruction = {'flat':flat,
                               'linear':minmod}
                               

    def swich_direction(self, A):
        """
        Synopsis
        --------
        Method for swiching the axies of array
        A. Where A is the state vector of either 
        primative or conserved variables.

        This method will be called going from i->j, 
        j->k and finally k->i as the primary direction 
        of integration.

        Args
        ----
        A: numpy-array like
        State vactor of fluid variables.
        """

        np.swapaxes(A, 2, self.direction_count+3)
        self.direction_count += 1
        return

    def reset_direction(self, A):
        """
        Synopsis
        --------
        Method for reseting the axies of array
        A. Where A is the state vector of either 
        primative or conserved variables.

        This method will set the axes of the velocities 
        to be {ijk} after the final call to axes switch 
        and before writting out to disk.

        Args
        ----
        A: numpy-array like
        State vactor of fluid variables.
        """
        if 



