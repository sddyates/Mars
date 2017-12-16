import numpy as np 

class Grid:

    def __init__(self, p):

        self.x1min = p['x1 min']
        self.x1max = p['x1 max']

        self.nx1 = p['resolution x1']

        self.dx1 = (abs(self.x1min) + abs(self.x1max))/self.nx1

        if p['reconstruction'] == 'flat':
            self.ghost_zones = 1
        elif p['reconstruction'] == 'linear':
            self.ghost_zones = 2
        elif p['reconstruction'] == 'pbolic':
            self.ghost_zones = 3

        self.ibeg = self.ghost_zones
        self.iend = self.nx1 + self.ghost_zones 

        self.lower_bc_ibeg = 0
        self.lower_bc_iend = self.ghost_zones - 1

        self.upper_bc_ibeg = self.nx1 + self.ghost_zones
        self.upper_bc_iend = self.nx1 + 2*self.ghost_zones - 1 

        self.imax = self.upper_bc_iend

        self.x1 = self._x1()

        if p['Dimensions'] == '1D':

            if p['method'] == 'hydro':
                self.nvar = 3
            elif p['method'] == 'mhd':
                self.nvar = 4

            self.shape_internal = [self.nvar, self.nx1]
            self.shape_flux_x1 = [self.nvar, self.nx1 + 1]

        if p['Dimensions'] == '2D':

            self.x2min = p['x2 min']
            self.x2max = p['x2 max']

            self.nx2 = p['resolution x2']

            self.dx2 = (abs(self.x2min) + abs(self.x2max))/self.nx2

            self.da = self.dx1*self.dx2

            self.jbeg = self.ghost_zones

            self.jend = self.nx2 + self.ghost_zones 

            self.lower_bc_jbeg = 0
 
            self.lower_bc_jend = self.ghost_zones - 1

            self.upper_bc_jbeg = self.nx2 + self.ghost_zones

            self.upper_bc_jend = self.nx2 + 2*self.ghost_zones - 1

            self.jmax = self.upper_bc_jend

            if p['method'] == 'hydro':
                self.nvar = 4
            elif p['method'] == 'mhd':
                self.nvar = 6

            self.shape_internal = [self.nvar, self.nx2, self.nx1]
            self.shape_flux_x2 = [self.nvar, self.nx2 + 1]

            self.x1 = self._x1()
            self.x2 = self._x2()

    def _x1(self):
        a = self.x1min - self.dx1*self.ghost_zones
        b = self.x1max + self.dx1*self.ghost_zones
        c = self.nx1 + 2*self.ghost_zones
        return np.linspace(a, b, c)

    def _x2(self):
        a = self.x2min - self.dx2*self.ghost_zones
        b = self.x2max + self.dx2*self.ghost_zones
        c = self.nx2 + 2*self.ghost_zones
        return np.linspace(a, b, c)

    def state_vector(self, p):
        if p['Dimensions'] == '1D':
            return np.zeros((self.nvar, 
                             2*self.ghost_zones + self.nx1))
        if p['Dimensions'] == '2D':
            return np.zeros((self.nvar, 
                             2*self.ghost_zones + self.nx2, 
                             2*self.ghost_zones + self.nx1))

    def boundary(self, U, p):
        self.LowerXBC(U, p['lower x1 boundary'], p['Dimensions'])
        self.UpperXBC(U, p['upper x1 boundary'], p['Dimensions'])
        if p['Dimensions'] == '2D':
            self.LowerYBC(U, p['lower x2 boundary'], p['Dimensions'])
            self.UpperYBC(U, p['upper x2 boundary'], p['Dimensions'])

    def LowerXBC(self, U, bc_type, dim):

        if bc_type == 'reciprocal' and dim == '1D':
            U[:, :self.ghost_zones] = \
                U[:, self.nx1:self.nx1 + self.ghost_zones] 

        if bc_type == 'outflow' and dim == '1D':
            U[:, :self.ghost_zones] = \
                U[:, self.ghost_zones + 1].reshape((3, 1))

        if bc_type == 'reciprocal' and dim == '2D':
            U[:, :, :self.ghost_zones] = \
                U[:, :, self.nx1:self.nx1 + self.ghost_zones] 


    def UpperXBC(self, U, bc_type, dim):

        if bc_type == 'reciprocal' and dim == '1D':
            U[:, self.upper_bc_ibeg:] = \
                U[:, self.ghost_zones:self.ghost_zones + 1] 

        if bc_type == 'outflow' and dim == '1D':
            U[:, self.upper_bc_ibeg:] = \
                U[:, self.upper_bc_ibeg - 1].reshape((3, 1))

        if bc_type == 'reciprocal' and dim == '2D':
            U[:, :, self.upper_bc_ibeg:] = \
                U[:, :, self.ghost_zones:self.ghost_zones + 1] 


    def LowerYBC(self, U, bc_type, dim):

        if bc_type == 'reciprocal' and dim == '2D':
            U[:, :self.ghost_zones, :] = \
                U[:, self.nx2:self.nx2 + self.ghost_zones, :] 


    def UpperYBC(self, U, bc_type, dim):

        if bc_type == 'reciprocal' and dim == '2D':
            U[:, self.upper_bc_ibeg:, :] = \
                U[:, self.ghost_zones:self.ghost_zones + 1, :] 





