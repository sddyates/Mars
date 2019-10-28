
import sys
import numpy as np

from riemann_solvers import tvdlf, hll, hllc
from cython_lib.riemann_solvers import tvdlf_pyx, hll_pyx, hllc_pyx
from reconstruction import flat, minmod
from cython_lib.reconstruction import flat_pyx, minmod_pyx
from time_stepping import Euler, RungaKutta2


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

    def __init__(self, p, l):
        self.assign_riemann_solver_(p)
        self.assign_reconstruction_(p)
        self.assign_time_stepping_(p)
        self.is_1D = p['Dimensions'] == '1D'
        self.is_2D = p['Dimensions'] == '2D'
        self.is_3D = p['Dimensions'] == '3D'
        self.gamma = np.float64(p['gamma'])
        self.gamma_1 = np.float64(self.gamma - 1.0)
        self.igamma_1 = 1.0/self.gamma_1
        self.cfl = np.float64(p['cfl'])

    def assign_riemann_solver_(self, p):
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


    def assign_reconstruction_(self, p):
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

    def assign_time_stepping_(self, p):
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
        else:
            print('Error: Invalid integrator.')
            sys.exit()
        return

    def assign_boundary_conditions(self, p):
        if p['Dimensions'] == '1D' and p['']:
            g.lowerX1BC(V, p['lower x1 boundary'], p['Dimensions'])
            g.upperX1BC(V, p['upper x1 boundary'], p['Dimensions'])
        elif p['Dimensions'] == '2D':
            g.lowerX1BC(V, p['lower x1 boundary'], p['Dimensions'])
            g.upperX1BC(V, p['upper x1 boundary'], p['Dimensions'])
            g.lowerX2BC(V, p['lower x2 boundary'], p['Dimensions'])
            g.upperX2BC(V, p['upper x2 boundary'], p['Dimensions'])
        elif p['Dimensions'] == '3D':
            g.lowerX1BC(V, p['lower x1 boundary'], p['Dimensions'])
            g.upperX1BC(V, p['upper x1 boundary'], p['Dimensions'])
            g.lowerX2BC(V, p['lower x2 boundary'], p['Dimensions'])
            g.upperX2BC(V, p['upper x2 boundary'], p['Dimensions'])
            g.lowerX3BC(V, p['lower x3 boundary'], p['Dimensions'])
            g.upperX3BC(V, p['upper x3 boundary'], p['Dimensions'])
        else:
            print('Error, invalid number of dimensions.')
            sys.exit()
