
import sys
import numpy as np

from riemann_solvers import tvdlf, hll, hllc
#  from cython_lib.riemann_solvers import tvdlf_pyx, hll_pyx, hllc_pyx
from reconstruction import flat, minmod
# from cython_lib.reconstruction import flat_pyx, minmod_pyx
from time_stepping import Euler, RungaKutta2, RungaKutta3


class Algorithm:
    """
    Synopsis
    --------
    This class allows for different algorithm components
    to be used and assigned at runtime without the need
    for if statments in the integration loop.

    Args
    ----
    p: dictionary-like
    dictionary of problem parameters.
    """

    def __init__(self, p):
        self._assign_riemann_solver(p)
        self._assign_reconstruction(p)
        self._assign_time_stepping(p)
        self.is_1D = p['Dimensions'] == '1D'
        self.is_2D = p['Dimensions'] == '2D'
        self.is_3D = p['Dimensions'] == '3D'
        self.gamma = np.float64(p['gamma'])
        self.gamma_1 = np.float64(self.gamma - 1.0)
        self.igamma_1 = 1.0/self.gamma_1
        self.smapp_pressure = 1.0e-12


    def _assign_riemann_solver(self, p):
        """
        Synopsis
        --------
        This method assigns the function call for
        the Riemann solver.

        Args
        ----
        p: dictionary-like
        dictionary of problem parameters.
        """
        if (p['riemann'] == 'tvdlf') & (p['optimisation'] == 'numba'):
            self.riemann_solver = tvdlf
        elif (p['riemann'] == 'hll') & (p['optimisation'] == 'numba'):
            self.riemann_solver = hll
        elif (p['riemann'] == 'hllc') & (p['optimisation'] == 'numba'):
            self.riemann_solver = hllc
        elif (p['riemann'] == 'tvdlf') & (p['optimisation'] == 'cython'):
            self.riemann_solver = tvdlf_pyx
        elif (p['riemann'] == 'hll') & (p['optimisation'] == 'cython'):
            self.riemann_solver = hll_pyx
        elif (p['riemann'] == 'hllc') & (p['optimisation'] == 'cython'):
            self.riemann_solver = hllc_pyx
        else:
            print('Error: invalid riennman solver.')
            sys.exit()


    def _assign_reconstruction(self, p):
        """
        Synopsis
        --------
        This method assigns the function call for
        the reconstruction stage.

        Args
        ----
        p: dictionary-like
        dictionary of problem parameters.
        """
        if (p['reconstruction'] == 'flat') & (p['optimisation'] == 'numba'):
            self.reconstruction = flat
        elif (p['reconstruction'] == 'linear') & (p['optimisation'] == 'numba'):
            self.reconstruction = minmod
        elif (p['reconstruction'] == 'flat') & (p['optimisation'] == 'cython'):
            self.reconstruction = flat_pyx
        elif (p['reconstruction'] == 'linear') & (p['optimisation'] == 'cython'):
            self.reconstruction = minmod_pyx
        else:
            print('Error: Invalid reconstructor.')
            sys.exit()
        return


    def _assign_time_stepping(self, p):
        """
        Synopsis
        --------
        This method assigns the function call for
        the method used for time stepping.

        Args
        ----
        p: dictionary-like
        dictionary of problem parameters.
        """
        if p['time stepping'] == 'Euler':
            self.time_incriment = Euler
        elif p['time stepping'] == 'RK2':
            self.time_incriment = RungaKutta2
        elif p['time stepping'] == 'RK3':
            self.time_incriment = RungaKutta3
        else:
            print('Error: Invalid integrator.')
            sys.exit()
        return
