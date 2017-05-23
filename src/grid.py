import numpy as np 

class Grid:

    def __init__(self, para):
        self.xmin = para['xmin']
        self.ymin = para['ymin']
        self.xmax = para['xmax']
        self.ymax = para['xmax']
        self.nx = para['nx']
        self.ny = para['ny']

        if para['reconstruction'] == 'flat':
            self.nxg = 1
            self.nyg = 1
        if para['reconstruction'] == 'linear':
            self.nxg = 2
            self.nyg = 2
        elif para['reconstruction'] == 'parabolic':
            self.nxg = 3
            self.nyg = 3

        if para['method'] == 'hydro':
            self.nvar = 4
        elif para['method'] == 'mhd':
            self.nvar = 6

    def imax(self):
        return self.upper_bc_iend()

    def jmax(self):
        return self.upper_bc_jend()

    def x(self):
        a = self.xmin - self.dx()*self.nxg
        b = self.xmax + self.dx()*self.nxg
        c = self.nx + 2*self.nxg
        return np.linspace(a, b, c)

    def y(self):
        a = self.ymin - self.dy()*self.nyg
        b = self.ymax + self.dy()*self.nyg
        c = self.ny + 2*self.nyg
        return np.linspace(a, b, c)

    def state_vector(self):
        x_extent = 2*self.nxg + self.nx
        y_extent = 2*self.nyg + self.ny
        return np.zeros((self.nvar, x_extent, y_extent))

    def dx(self):
        return (abs(self.xmin) + abs(self.xmax))/self.nx

    def dy(self):
        return (abs(self.ymin) + abs(self.ymax))/self.ny

    def ibeg(self):
        return self.nxg

    def jbeg(self):
        return self.nyg

    def iend(self):
        return self.nx + self.nxg 

    def jend(self):
        return self.ny + self.nyg 

    def lower_bc_ibeg(self):
        return 0

    def lower_bc_jbeg(self):
        return 0

    def lower_bc_iend(self):
        return self.nxg - 1

    def lower_bc_jend(self):
        return self.nyg - 1

    def upper_bc_ibeg(self):
        return self.nx + self.nxg

    def upper_bc_jbeg(self):
        return self.ny + self.nyg

    def upper_bc_iend(self):
        return self.nx + 2*self.nxg - 1 

    def upper_bc_jend(self):
        return self.ny + 2*self.nyg - 1

class Boundary:

    def __init__(self, a, g, para):
        self.LowerXBC(a, g, para['lower_bc_x'])
        self.UpperXBC(a, g, para['upper_bc_x'])
        self.LowerYBC(a, g, para['lower_bc_y'])
        self.UpperYBC(a, g, para['upper_bc_y'])

    def LowerXBC(self, a, g, bc_type):
        if bc_type == 'reciprocal':
            a[:, :g.nxg, g.jbeg():g.jend()] = \
                a[:, g.nx:g.nx+g.nxg, g.jbeg():g.jend()] 

    def UpperXBC(self, a, g, bc_type):
        if bc_type == 'reciprocal':
            a[:, g.upper_bc_ibeg():, g.jbeg():g.jend()] = \
                a[:, g.nxg:g.nxg+1, g.jbeg():g.jend()] 

    def LowerYBC(self, a, g, bc_type):
        if bc_type == 'reciprocal':
            a[:, g.ibeg():g.iend(), :g.nyg] = \
                a[:, g.ibeg():g.iend(), g.ny:g.ny+g.nyg] 

    def UpperYBC(self, a, g, bc_type):
        if bc_type == 'reciprocal':
            a[:, g.ibeg():g.iend(), g.upper_bc_jbeg():] = \
                a[:, g.ibeg():g.iend(), g.nyg:g.nyg+1] 







