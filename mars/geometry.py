import numpy as np
from solvers import tvdlf, hll, hllc
from piecewise import flat, minmod

class Geometry:

    def __init__(self, p):
        self.direction_count = 0

    def swich_to_j_from_i(self, A, case):
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

    def swich_to_k_from_j(self, A):
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

    def swich_to_i_from_k(self, A):
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



