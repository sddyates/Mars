
import sys
from solvers import tvdlf, hll, hllc
from piecewise import flat, minmod

class Algorithm:

    def __init__(self, p):
        self.assign_riemann_solver(p)
        self.assign_reconstruction(p)

    def assign_riemann_solver(self, p):

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
        if p['reconstruction'] == 'flat':
            self.reconstruction = flat
        elif p['reconstruction'] == 'linear':
            self.reconstruction = minmod
        else:
            print('Error: Invalid reconstructor.')
            sys.exit()