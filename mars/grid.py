
import numpy as np
import sys


class Grid:

    """
    Synopsis
    --------
    Construct variables and structures that define the grid
    and vectors for the variables and dimensions.

    Args
    ----
    p: dic-like
    Dictionary of user defined parameters, e.g.
    maximum simulation time.

    Attributes
    ----------
    state_vector()
    Define the vector structure for the variables.

    boundary()
    Assignes the boundary condtions.

    TODO
    ----
    Expand the definitions to 3D.
    """

    def __init__(self, p):

        self.speed_max = 0.0
        self.cs_max = 0.0

        if p['reconstruction'] == 'flat':
            self.gz = 1
        elif p['reconstruction'] == 'linear':
            self.gz = 2
        elif p['reconstruction'] == 'parabolic':
            self.gz = 3

        if p['Dimensions'] == '1D':

            self.x1min = p['x1 min']
            self.x1max = p['x1 max']

            self.nx1 = p['resolution x1']
            self.rez = self.nx1

            self.dx1 = (abs(self.x1min) + abs(self.x1max))/self.nx1
            self.dxi = [self.dx1]
            self.min_dxi = np.amin(self.dxi)

            self.ibeg = self.gz
            self.iend = self.nx1 + self.gz

            self.lower_bc_ibeg = 0
            self.lower_bc_iend = self.gz - 1

            self.upper_bc_ibeg = self.nx1 + self.gz
            self.upper_bc_iend = self.nx1 + 2*self.gz - 1

            self.imax = self.upper_bc_iend

            if p['method'] == 'hydro':
                self.nvar = 3
            elif p['method'] == 'mhd':
                self.nvar = 4

            self.shape_internal = [self.nvar, self.nx1]
            self.shape_flux_x1 = [self.nvar, self.nx1 + 1]

            self.shape_flux = [self.nvar, self.nx1 + 1]

            self.x1 = self._x1()

            self.x1_verts = self._x1_verts()

        if p['Dimensions'] == '2D':

            self.x1min = p['x1 min']
            self.x1max = p['x1 max']
            self.x2min = p['x2 min']
            self.x2max = p['x2 max']

            self.nx1 = p['resolution x1']
            self.nx2 = p['resolution x2']
            self.rez = self.nx1*self.nx2

            self.dx1 = (abs(self.x1min) + abs(self.x1max))/self.nx1
            self.dx2 = (abs(self.x2min) + abs(self.x2max))/self.nx2
            self.dxi = [self.dx1, self.dx2]
            self.min_dxi = np.amin(self.dxi)

            self.da = self.dx1*self.dx2

            self.ibeg = self.gz
            self.iend = self.nx1 + self.gz
            self.jbeg = self.gz
            self.jend = self.nx2 + self.gz

            self.lower_bc_ibeg = 0
            self.lower_bc_iend = self.gz - 1
            self.lower_bc_jbeg = 0
            self.lower_bc_jend = self.gz - 1

            self.upper_bc_ibeg = self.nx1 + self.gz
            self.upper_bc_iend = self.nx1 + 2*self.gz - 1
            self.upper_bc_jbeg = self.nx2 + self.gz
            self.upper_bc_jend = self.nx2 + 2*self.gz - 1

            self.jmax = self.upper_bc_jend

            if p['method'] == 'hydro':
                self.nvar = 4
            elif p['method'] == 'mhd':
                self.nvar = 6

            self.shape_internal = [self.nvar, self.nx2, self.nx1]
            self.shape_flux_x1 = [self.nvar, self.nx1 + 1]
            self.shape_flux_x2 = [self.nvar, self.nx2 + 1]

            self.shape_flux = [[self.nvar, self.nx1 + 1],
                               [self.nvar, self.nx2 + 1]]

            self.x1 = self._x1()
            self.x2 = self._x2()

            self.x1_verts = self._x1_verts()
            self.x2_verts = self._x2_verts()

        if p['Dimensions'] == '3D':

            self.x1min = p['x1 min']
            self.x1max = p['x1 max']
            self.x2min = p['x2 min']
            self.x2max = p['x2 max']
            self.x3min = p['x3 min']
            self.x3max = p['x3 max']

            self.nx1 = p['resolution x1']
            self.nx2 = p['resolution x2']
            self.nx3 = p['resolution x3']
            self.rez = self.nx1*self.nx2*self.nx3

            self.dx1 = (abs(self.x1min) + abs(self.x1max))/self.nx1
            self.dx2 = (abs(self.x2min) + abs(self.x2max))/self.nx2
            self.dx3 = (abs(self.x3min) + abs(self.x3max))/self.nx3
            self.dxi = [self.dx1, self.dx2, self.dx3]
            self.min_dxi = np.amin(self.dxi)

            self.dv = self.dx1*self.dx2*self.dx3

            self.ibeg = self.gz
            self.iend = self.nx1 + self.gz
            self.jbeg = self.gz
            self.jend = self.nx2 + self.gz
            self.kbeg = self.gz
            self.kend = self.nx3 + self.gz

            self.lower_bc_ibeg = 0
            self.lower_bc_iend = self.gz - 1
            self.lower_bc_jbeg = 0
            self.lower_bc_jend = self.gz - 1
            self.lower_bc_kbeg = 0
            self.lower_bc_kend = self.gz - 1

            self.upper_bc_ibeg = self.nx1 + self.gz
            self.upper_bc_iend = self.nx1 + 2*self.gz - 1
            self.upper_bc_jbeg = self.nx2 + self.gz
            self.upper_bc_jend = self.nx2 + 2*self.gz - 1
            self.upper_bc_kbeg = self.nx3 + self.gz
            self.upper_bc_kend = self.nx3 + 2*self.gz - 1

            self.imax = self.upper_bc_iend
            self.jmax = self.upper_bc_jend
            self.kmax = self.upper_bc_kend

            if p['method'] == 'hydro':
                self.nvar = 5
            elif p['method'] == 'mhd':
                self.nvar = 8

            self.shape_internal = [self.nvar, self.nx3, self.nx2, self.nx1]
            self.shape_flux_x1 = [self.nvar, self.nx1 + 1]
            self.shape_flux_x2 = [self.nvar, self.nx2 + 1]
            self.shape_flux_x3 = [self.nvar, self.nx3 + 1]

            self.shape_flux = [[self.nvar, self.nx1 + 1],
                               [self.nvar, self.nx2 + 1],
                               [self.nvar, self.nx3 + 1]]

            self.x1 = self._x1()
            self.x2 = self._x2()
            self.x3 = self._x3()

            #self.x1, self.x2, self.x3 = np.meshgrid(self._x1(),
            #                                        self._x2(),
            #                                        self._X3(),
            #                                        sparse=False,
            #                                        indexing='ij')

            self.x1_verts = self._x1_verts()
            self.x2_verts = self._x2_verts()
            self.x3_verts = self._x3_verts()

    def _x1(self):
        a = self.x1min - self.dx1*self.gz
        b = self.x1max + self.dx1*self.gz
        c = self.nx1 + 2*self.gz
        return np.linspace(a, b, c)

    def _x1_verts(self):
        a = self.x1 - self.dx1/2.0
        b = self.x1[-1] + self.dx1/2.0
        return np.append(a, b)

    def _x2(self):
        a = self.x2min - self.dx2*self.gz
        b = self.x2max + self.dx2*self.gz
        c = self.nx2 + 2*self.gz
        return np.linspace(a, b, c)

    def _x2_verts(self):
        a = self.x2 - self.dx2/2.0
        b = self.x2[-1] + self.dx2/2.0
        return np.append(a, b)

    def _x3(self):
        a = self.x3min - self.dx3*self.gz
        b = self.x3max + self.dx3*self.gz
        c = self.nx3 + 2*self.gz
        return np.linspace(a, b, c)

    def _x3_verts(self):
        a = self.x3 - self.dx3/2.0
        b = self.x3[-1] + self.dx3/2.0
        return np.append(a, b)

    def state_vector(self, p):
        if p['Dimensions'] == '1D':
            return np.zeros((self.nvar,
                             2*self.gz + self.nx1))
        elif p['Dimensions'] == '2D':
            return np.zeros((self.nvar,
                             2*self.gz + self.nx2,
                             2*self.gz + self.nx1))
        elif p['Dimensions'] == '3D':
            return np.zeros((self.nvar,
                             2*self.gz + self.nx3,
                             2*self.gz + self.nx2,
                             2*self.gz + self.nx1))
        else:
            print('Error, invalid number of dimensions.')
            sys.exit()

    def build_fluxes(self, vxn):
        if vxn == 2:
            array_shape = self.shape_flux[0]
        if vxn == 3:
            array_shape = self.shape_flux[1]
        if vxn == 4:
            array_shape = self.shape_flux[2]

        self.flux = np.zeros(shape=array_shape)
        self.FL = np.zeros(shape=array_shape)
        self.FR = np.zeros(shape=array_shape)
        self.UL = np.zeros(shape=array_shape)
        self.UR = np.zeros(shape=array_shape)
        self.VL = np.zeros(shape=array_shape)
        self.VR = np.zeros(shape=array_shape)
        self.SL = np.zeros(shape=array_shape[1])
        self.SR = np.zeros(shape=array_shape[1])
        self.pres = np.zeros(shape=array_shape[1])

    def boundary(self, V, p):

        if p['Dimensions'] == '1D':
            self._lowerX1BC(V, p['lower x1 boundary'], p['Dimensions'])
            self._upperX1BC(V, p['upper x1 boundary'], p['Dimensions'])
        elif p['Dimensions'] == '2D':
            self._lowerX1BC(V, p['lower x1 boundary'], p['Dimensions'])
            self._upperX1BC(V, p['upper x1 boundary'], p['Dimensions'])
            self._lowerX2BC(V, p['lower x2 boundary'], p['Dimensions'])
            self._upperX2BC(V, p['upper x2 boundary'], p['Dimensions'])
        elif p['Dimensions'] == '3D':
            self._lowerX1BC(V, p['lower x1 boundary'], p['Dimensions'])
            self._upperX1BC(V, p['upper x1 boundary'], p['Dimensions'])
            self._lowerX2BC(V, p['lower x2 boundary'], p['Dimensions'])
            self._upperX2BC(V, p['upper x2 boundary'], p['Dimensions'])
            self._lowerX3BC(V, p['lower x3 boundary'], p['Dimensions'])
            self._upperX3BC(V, p['upper x3 boundary'], p['Dimensions'])
        else:
            print('Error, invalid number of dimensions.')
            sys.exit()

    def _lowerX1BC(self, V, bc_type, dim):

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

            for o in range(self.gz):
                V[:, :, o] = V[:, :, self.gz]

            #V[:, :, :self.gz] = \
            #    V[:, :, self.gz]#.reshape(
                #(self.nvar,
                # 2*self.gz+self.nx2,
                # self.gz-1))

        elif bc_type == 'reciprocal' and dim == '3D':
            V[:, :, :, :self.gz] = \
                V[:, :, :, self.nx1:self.nx1 + self.gz]

        elif bc_type == 'outflow' and dim == '3D':
            V[:, :, :, :self.gz] = \
                V[:, :, :, self.gz].reshape(
                    (self.nvar,
                     2*self.gz+self.nx3,
                     2*self.gz+self.nx2,
                     self.gz-1))

        else:
            print('Error, invalid lower x1 boundary.')
            sys.exit()

    def _upperX1BC(self, V, bc_type, dim):

        if bc_type == 'reciprocal' and dim == '1D':
            V[:, self.upper_bc_ibeg:] = \
                V[:, self.gz:self.gz + 1]

        elif bc_type == 'outflow' and dim == '1D':
            V[:, self.upper_bc_ibeg:] = \
                V[:, self.upper_bc_ibeg - 1].reshape(
                    (self.nvar,
                     self.gz-1))

        elif bc_type == 'reciprocal' and dim == '2D':
            V[:, :, self.upper_bc_ibeg:] = \
                V[:, :, self.gz:self.gz + 1]

        elif bc_type == 'outflow' and dim == '2D':
            for o in range(self.upper_bc_ibeg, self.upper_bc_iend+1):
                V[:, :, o] = V[:, :, self.upper_bc_ibeg - 1]

            #V[:, :, self.upper_bc_ibeg:] = \
            #    V[:, :, self.upper_bc_ibeg - 1].reshape(
            #        (self.nvar,
            #         2*self.gz+self.nx2,
            #         self.gz - 1))

        elif bc_type == 'reciprocal' and dim == '3D':
            V[:, :, :, self.upper_bc_ibeg:] = \
                V[:, :, :, self.gz:self.gz + 1]

        elif bc_type == 'outflow' and dim == '3D':
            V[:, :, :, self.upper_bc_ibeg:] = \
                V[:, :, :, self.upper_bc_ibeg - 1].reshape(
                    (self.nvar,
                     2*self.gz+self.nx3,
                     2*self.gz+self.nx2,
                     self.gz - 1))

        else:
            print('Error, invalid upper x1 boundary.')
            sys.exit()

    def _lowerX2BC(self, V, bc_type, dim):

        if bc_type == 'reciprocal' and dim == '2D':
            V[:, :self.gz, :] = \
                V[:, self.nx2:self.nx2 + self.gz, :]

        elif bc_type == 'outflow' and dim == '2D':

            for o in range(self.gz):
                V[:, o, :] = V[:, self.gz, :]

            #V[:, :self.gz, :] = \
            #    V[:, self.gz, :].reshape(
            #        (self.nvar,
            #         self.gz - 1,
            #         self.nx1 + 2*self.gz))

        elif bc_type == 'reciprocal' and dim == '3D':
            V[:, :, :self.gz, :] = \
                V[:, :, self.nx2:self.nx2 + self.gz, :]

        elif bc_type == 'outflow' and dim == '3D':
            V[:, :, :self.gz, :] = \
                V[:, :, self.gz, :].reshape(
                    (self.nvar,
                     2*self.gz+self.nx3,
                     self.gz - 1,
                     2*self.gz+self.nx1))

        else:
            print('Error, invalid lower x2 boundary.')
            sys.exit()

    def _upperX2BC(self, V, bc_type, dim):

        if bc_type == 'reciprocal' and dim == '2D':
            V[:, self.upper_bc_jbeg:, :] = \
                V[:, self.gz:self.gz + 1, :]

        elif bc_type == 'outflow' and dim == '2D':

            for o in range(self.upper_bc_ibeg, self.upper_bc_iend+1):
                V[:, o, :] = V[:, self.upper_bc_ibeg - 1, :]

            #V[:, self.upper_bc_jbeg:, :] = \
            #    V[:, self.upper_bc_jbeg - 1, :].reshape(
            #        (self.nvar,
            #         self.gz - 1,
            #         self.nx1 + 2*self.gz))

        elif bc_type == 'reciprocal' and dim == '3D':
            V[:, :, self.upper_bc_jbeg:, :] = \
                V[:, :, self.gz:self.gz + 1, :]

        elif bc_type == 'outflow' and dim == '3D':
            V[:, :, self.upper_bc_jbeg:, :] = \
                V[:, :, self.upper_bc_jbeg - 1, :].reshape(
                    (self.nvar,
                     2*self.gz+self.nx3,
                     self.gz - 1,
                     2*self.gz+self.nx1))

        else:
            print('Error, invalid upper x2 boundary.')
            sys.exit()

    def _lowerX3BC(self, V, bc_type, dim):

        if bc_type == 'reciprocal' and dim == '3D':
            V[:, :self.gz, :, :] = \
                V[:, self.nx3:self.nx3 + self.gz, :, :]

        elif bc_type == 'outflow' and dim == '3D':
            V[:, :self.gz, :, :] = \
                V[:, self.gz, :, :].reshape(
                    (self.nvar,
                     self.gz - 1,
                     2*self.gz+self.nx2,
                     2*self.gz+self.nx1,))

        else:
            print('Error, invalid upper x3 boundary.')
            sys.exit()

    def _upperX3BC(self, V, bc_type, dim):

        if bc_type == 'reciprocal' and dim == '3D':
            V[:, self.upper_bc_kbeg:, :, :] = \
                V[:, self.gz:self.gz + 1, :, :]

        elif bc_type == 'outflow' and dim == '3D':
            V[:, self.upper_bc_kbeg:, :, :] = \
                V[:, self.upper_bc_kbeg - 1, :, :].reshape(
                    (self.nvar,
                     self.gz - 1,
                     2*self.gz+self.nx2,
                     2*self.gz+self.nx1,))

        else:
            print('Error, invalid upper x3 boundary.')
            sys.exit()
