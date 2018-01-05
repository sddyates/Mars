import numpy as np 
from globe import *
import sys

class Grid:

    def __init__(self, p):

        self.x1min = p['x1 min']
        self.x1max = p['x1 max']

        self.nx1 = p['resolution x1']

        self.dx1 = (abs(self.x1min) + abs(self.x1max))/self.nx1

        if p['reconstruction'] == 'flat':
            self.gz = 1
        elif p['reconstruction'] == 'linear':
            self.gz = 2
        elif p['reconstruction'] == 'parabolic':
            self.gz = 3

        self.ibeg = self.gz
        self.iend = self.nx1 + self.gz 

        self.lower_bc_ibeg = 0
        self.lower_bc_iend = self.gz - 1

        self.upper_bc_ibeg = self.nx1 + self.gz
        self.upper_bc_iend = self.nx1 + 2*self.gz - 1 

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

            self.jbeg = self.gz

            self.jend = self.nx2 + self.gz 

            self.lower_bc_jbeg = 0
 
            self.lower_bc_jend = self.gz - 1

            self.upper_bc_jbeg = self.nx2 + self.gz

            self.upper_bc_jend = self.nx2 + 2*self.gz - 1

            self.jmax = self.upper_bc_jend

            if p['method'] == 'hydro':
                self.nvar = 4
            elif p['method'] == 'mhd':
                self.nvar = 6

            self.shape_internal = [self.nvar, self.nx2, self.nx1]
            self.shape_flux_x2 = [self.nvar, self.nx2 + 1]
            self.shape_flux_x1 = [self.nvar, self.nx1 + 1]

            self.x1 = self._x1()
            self.x2 = self._x2()

    def _x1(self):
        a = self.x1min - self.dx1*self.gz
        b = self.x1max + self.dx1*self.gz
        c = self.nx1 + 2*self.gz
        return np.linspace(a, b, c)

    def _x2(self):
        a = self.x2min - self.dx2*self.gz
        b = self.x2max + self.dx2*self.gz
        c = self.nx2 + 2*self.gz
        return np.linspace(a, b, c)

    def state_vector(self, p):
        if p['Dimensions'] == '1D':
            return np.zeros((self.nvar, 
                             2*self.gz + self.nx1))
        if p['Dimensions'] == '2D':
            return np.zeros((self.nvar, 
                             2*self.gz + self.nx2, 
                             2*self.gz + self.nx1))

    def boundary(self, V, p):

        if p['Dimensions'] == '1D':
            self._lowerXBC(V, p['lower x1 boundary'], p['Dimensions'])
            self._upperXBC(V, p['upper x1 boundary'], p['Dimensions'])
        elif p['Dimensions'] == '2D':
            self._lowerXBC(V, p['lower x1 boundary'], p['Dimensions'])
            self._upperXBC(V, p['upper x1 boundary'], p['Dimensions'])
            self._lowerYBC(V, p['lower x2 boundary'], p['Dimensions'])
            self._upperYBC(V, p['upper x2 boundary'], p['Dimensions'])
        else:
            print('Error, invalid number of dimensions.')
            sys.exit()

    def _lowerXBC(self, V, bc_type, dim):

        if bc_type == 'reciprocal' and dim == '1D':
            V[:, :self.gz] = \
                V[:, self.nx1:self.nx1 + self.gz] 

        elif bc_type == 'outflow' and dim == '1D':
            V[:, :self.gz] = \
                V[:, self.gz].reshape(self.nvar, self.gz-1)

        elif bc_type == 'reciprocal' and dim == '2D':
            V[:, :, :self.gz] = \
                V[:, :, self.nx1:self.nx1 + self.gz] 

        elif bc_type == 'outflow' and dim == '2D':
            V[:, :, :self.gz] = \
                V[:, :, self.gz].reshape(
                    self.nvar, 2*self.gz+self.nx2, self.gz - 1)

        else:
            print('Error, invalid lower x1 boundary.')
            sys.exit()


    def _upperXBC(self, V, bc_type, dim):

        if bc_type == 'reciprocal' and dim == '1D':
            V[:, self.upper_bc_ibeg:] = \
                V[:, self.gz:self.gz + 1] 

        if bc_type == 'outflow' and dim == '1D':
            V[:, self.upper_bc_ibeg:] = \
                V[:, self.upper_bc_ibeg - 1].reshape(self.nvar, self.gz-1)

        elif bc_type == 'reciprocal' and dim == '2D':
            V[:, :, self.upper_bc_ibeg:] = \
                V[:, :, self.gz:self.gz + 1] 

        elif bc_type == 'outflow' and dim == '2D':
            V[:, :, self.upper_bc_ibeg:] = \
                V[:, :, self.upper_bc_ibeg - 1].reshape(
                    (self.nvar, 2*self.gz+self.nx2, self.gz - 1))

        else:
            print('Error, invalid upper x1 boundary.')
            sys.exit()


    def _lowerYBC(self, V, bc_type, dim):

        if bc_type == 'reciprocal' and dim == '2D':
            V[:, :self.gz, :] = \
                V[:, self.nx2:self.nx2 + self.gz, :] 

        elif bc_type == 'outflow' and dim == '2D':
            V[:, :self.gz, :] = \
                V[:, self.gz, :].reshape(
                    (self.nvar, self.gz - 1, self.nx1 + 2*self.gz))

        else:
            print('Error, invalid lower x2 boundary.')
            sys.exit()


    def _upperYBC(self, V, bc_type, dim):

        if bc_type == 'reciprocal' and dim == '2D':
            V[:, self.upper_bc_jbeg:, :] = \
                V[:, self.gz:self.gz + 1, :] 

        elif bc_type == 'outflow' and dim == '2D':
            V[:, self.upper_bc_jbeg:, :] = \
                V[:, self.upper_bc_jbeg - 1, :].reshape(
                    (self.nvar, self.gz - 1, self.nx1 + 2*self.gz))

        else:
            print('Error, invalid upper x2 boundary.')
            sys.exit()



