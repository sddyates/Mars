
import sys
from riemann_solvers import tvdlf, hll, hllc
from reconstruction import flat, minmod
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

    def __init__(self, p):
        self.assign_riemann_solver(p)
        self.assign_reconstruction(p)
        self.assign_time_stepping(p)
        self.is_1D = p['Dimensions'] == '1D'
        self.is_2D = p['Dimensions'] == '2D'
        self.is_3D = p['Dimensions'] == '3D'
        self.gamma = p['gamma']
        self.gamma_1 = self.gamma - 1.0
        self.cfl = p['cfl']

    def assign_riemann_solver(self, p):
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
        if p['riemann'] == 'tvdlf':
            self.riemann_solver = tvdlf
        elif p['riemann'] == 'hll':
            self.riemann_solver = hll
        elif p['riemann'] == 'hllc':
            self.riemann_solver = hllc
        else:
            print('Error: invalid riennman solver.')
            sys.exit()

    def assign_reconstruction(self, p):
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
        if p['reconstruction'] == 'flat':
            self.reconstruction = flat
        elif p['reconstruction'] == 'linear':
            self.reconstruction = minmod
        else:
            print('Error: Invalid reconstructor.')
            sys.exit()


    def assign_time_stepping(self, p):
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