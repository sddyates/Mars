
import sys
import numpy as np
import numba as nb

from time_stepping import Euler, RungaKutta2, RungaKutta3
from riemann_solvers import tvdlf, hll, hllc
from reconstruction import flat, minmod
from cython_lib.riemann_solvers import tvdlf_pyx, hll_pyx, hllc_pyx
from cython_lib.reconstruction import flat_pyx, minmod_pyx

# spec = [
#     ('is_1D', nb.boolean),
#     ('is_2D', nb.boolean),
#     ('is_3D', nb.boolean),
#     ('p', ),
#     ('gamma', nb.float64),
#     ('gamma_1', nb.float64),
#     ('igamma_1', nb.float64),
#     ('small_pressure', nb.float64)
# ]

#@nb.jitclass(spec)
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

    def __init__(self, parameter):
        self._assign_riemann_solver(p)
        self._assign_reconstruction(p)
        self._assign_time_stepping(p)
        self.is_1D = parameter['Dimensions'] == '1D'
        self.is_2D = parameter['Dimensions'] == '2D'
        self.is_3D = parameter['Dimensions'] == '3D'
        self.gamma = np.float64(parameter['gamma'])
        self.gamma_1 = np.float64(self.gamma - 1.0)
        self.igamma_1 = 1.0/self.gamma_1
        self.small_pressure = 1.0e-12


    def _assign_riemann_solver(self, parameter):
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
        if (parameter['riemann'] == 'tvdlf') \
        & (parameter['optimisation'] == 'numba'):
            self.riemann_solver = tvdlf

        elif (parameter['riemann'] == 'hll') \
        & (parameter['optimisation'] == 'numba'):
            self.riemann_solver = hll

        elif (parameter['riemann'] == 'hllc') \
        & (parameter['optimisation'] == 'numba'):
            self.riemann_solver = hllc

        elif (parameter['riemann'] == 'tvdlf') \
        & (parameter['optimisation'] == 'cython'):
            self.riemann_solver = tvdlf_pyx

        elif (parameter['riemann'] == 'hll') \
        & (parameter['optimisation'] == 'cython'):
            self.riemann_solver = hll_pyx

        elif (parameter['riemann'] == 'hllc') \
        & (parameter['optimisation'] == 'cython'):
            self.riemann_solver = hllc_pyx

        else:
            print('Error: invalid riennman solver.')
            sys.exit()
        print(nb.typeof(self.riemann_solver))

        return


    def _assign_reconstruction(self, parameter):
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
        if (parameter['reconstruction'] == 'flat') \
        & (parameter['optimisation'] == 'numba'):
            self.reconstruction = flat

        elif (parameter['reconstruction'] == 'linear') \
        & (parameter['optimisation'] == 'numba'):
            self.reconstruction = minmod

        elif (parameter['reconstruction'] == 'flat') \
        & (parameter['optimisation'] == 'cython'):
            self.reconstruction = flat_pyx

        elif (parameter['reconstruction'] == 'linear') \
        & (parameter['optimisation'] == 'cython'):
            self.reconstruction = minmod_pyx

        else:
            print('Error: Invalid reconstructor.')
            sys.exit()

        return


    def _assign_time_stepping(self, parameter):
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
        if parameter['time stepping'] == 'Euler':
            self.time_incriment = Euler
        elif parameter['time stepping'] == 'RK2':
            self.time_incriment = RungaKutta2
        elif parameter['time stepping'] == 'RK3':
            self.time_incriment = RungaKutta3
        else:
            print('Error: Invalid integrator.')
            sys.exit()
        return


    def assign_boundary_conditions(self, parameter):
        if parameter['Dimensions'] == '1D' and parameter['']:
            g.lowerX1BC(
                V, parameter['lower x1 boundary'], parameter['Dimensions'])
            g.upperX1BC(
                V, parameter['upper x1 boundary'], parameter['Dimensions'])
        elif parameter['Dimensions'] == '2D':
            g.lowerX1BC(
                V, parameter['lower x1 boundary'], parameter['Dimensions'])
            g.upperX1BC(
                V, parameter['upper x1 boundary'], parameter['Dimensions'])
            g.lowerX2BC(
                V, parameter['lower x2 boundary'], parameter['Dimensions'])
            g.upperX2BC(
                V, parameter['upper x2 boundary'], parameter['Dimensions'])
        elif parameter['Dimensions'] == '3D':
            g.lowerX1BC(
                V, parameter['lower x1 boundary'], parameter['Dimensions'])
            g.upperX1BC(
                V, parameter['upper x1 boundary'], parameter['Dimensions'])
            g.lowerX2BC(
                V, parameter['lower x2 boundary'], parameter['Dimensions'])
            g.upperX2BC(
                V, parameter['upper x2 boundary'], parameter['Dimensions'])
            g.lowerX3BC(
                V, parameter['lower x3 boundary'], parameter['Dimensions'])
            g.upperX3BC(
                V, parameter['upper x3 boundary'], parameter['Dimensions'])
        else:
            print('Error, invalid number of dimensions.')
            sys.exit()
