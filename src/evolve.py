import numpy as np
from globe import *
import sys
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)

class Evolve:

    def __init__(self):
        None

    def incriment(self, U, dt, g, p):
        """
        Synopsis
        --------
        Evolve the simulation domain though one 
        time step.

        Args
        ----
        U: numpy array-like
            State vector containing the hole solution 
            and all variables

        dt: double-like
            Time step, in simulation units.

        g: object-like
            Object containing all variables related to 
            the grid, e.g. cell width.

        p: dic-like
            Dictionary of user defined ps, e.g. 
            maximum simulation time.        

        Attributes
        ----------
        None.

        TODO
        ----
        None
        """

        # Convert primitive to conservative variables.
        Q = self._prims_to_cons(U, p['gamma'], p['Dimensions'])

        # Time step with Euler.
        if p['time stepping'] == 'Euler':
            Q_new = Q + dt/g.dxi*self._flux(Q, g, p)

        
        # Time step with RK2.
        elif p['time stepping'] == 'RK2':

            # k1 step.
            Q_star = Q + 0.5*dt*self._flux(Q, g, p)

            # Reaply boundary conditions.
            g.boundary(Q_star, p)

            # k2 step.
            Q_new = Q_star + 0.5*dt*self._flux(Q_star, g, p)
        
        else:
            print('Error: Invalid integrator.')
            sys.exit()

        # Convert primitive to conservative variables.
        U = self._cons_to_prims(Q_new, p['gamma'], p['Dimensions'])

        if np.isnan(np.sum(U)):
            print("Error, nan in array, function: incriment")
            sys.exit()

        return U

    def _flux(self, Q, g, p):

        dflux_x1 = np.zeros(shape=Q.shape)

        if p['Dimensions'] == '1D':
            face_flux_x1 = self._riemann(Q, g, p, 'i')
            dflux_x1[:, g.ibeg:g.iend] = (face_flux_x1[:, :-1] - face_flux_x1[:, 1:])

            dflux = dflux_x1

        '''
        if p['Dimensions'] == '2D':
            dflux_x2 = np.zeros(shape=Q.shape)
            # Transvers loop over columns in x1.
            for j in range(g.jbeg, g.jend):
                face_flux_x1 = self._riemann(Q[:, j, :], g, p, 'i')
                dflux_x1[:, j, g.ibeg:g.iend] = -1.0/g.da*(g.dx2*face_flux_x1[:, :-1] - g.dx2*face_flux_x1[:, 1:])

            # Transvers loop over columns in x2.
            for i in range(g.ibeg, g.iend):                                       
                face_flux_x2 = self._riemann(Q[:, :, i], g, p, 'j')
                dflux_x2[:, g.jbeg:g.jend, i] = -1.0/g.da*(g.dx1*face_flux_x2[:, :-1] - g.dx1*face_flux_x2[:, 1:])

            dflux = dflux_x1 + dflux_x2
        '''

        if np.isnan(np.sum(dflux)):
            print("Error, nan in array, function: flux")
            sys.exit()

        return dflux


    def _reconstruction(self, y, g, p, axis):
        '''
                                  
                    G                         G
          ^  ___ ___|_____ ...________________|_______
        y | |   |   |   |     |   |   |   |   |   |   |
            |___|___|___|_ ...|___|___|___|___|___|___|
              0   1   2    ...   
                  i
                 _______ ...__________________
             L  |   |   |     |   |   |   |   |
                |___|___|...__|___|___|___|___|

                     _______ ...__________________
                  R |   |   |     |   |   |   |   |
                    |___|___|...__|___|___|___|___|

                   _______ ...__________________
             flux |   |   |     |   |   |   |   |
                  |___|___|...__|___|___|___|___|

        '''
        if axis == 'i':
            dxi = g.dx1
        if axis == 'j':
            dxi = g.dx2

        if p['reconstruction'] == 'flat':
            L = y[:, :-g.ghost_zones]
            R = y[:, g.ghost_zones:]

        elif p['reconstruction'] == 'linear':

            def linear(y, dxi, p, side):                

                if p['limiter'] == None:                    
                    m = np.gradient(y, dxi, axis=1)

                elif p['limiter'] == 'minmod':               
                    a = (y[:, 1:-1] - y[:, :-g.ghost_zones])/dxi 
                    b = (y[:, g.ghost_zones:] - y[:, 1:-1])/dxi
                    m = np.zeros(shape=a.shape)

                    # Pythonic witchcraft:
                    slicing1 = np.where((abs(a) < abs(b)) & (a*b > 0.0))
                    slicing2 = np.where((abs(a) > abs(b)) & (a*b > 0.0))
                    slicing3 = np.where(a*b < 0.0)
                    m[slicing1] = a[slicing1]
                    m[slicing2] = b[slicing2]
                    m[slicing3] = 0.0

                else:
                    print('Error: Invalid limiter.')
                    sys.exit()

                if side == 'left':
                    return y[:, g.ghost_zones - 1:-g.ghost_zones] \
                           + m[:, :-1]/2.0*dxi
                if side == 'right':
                    return y[:, g.ghost_zones:-g.ghost_zones + 1] \
                          - m[:, 1:]/2.0*dxi

            L = linear(y, dxi, p, 'left')
            R = linear(y, dxi, p, 'right')

        else:
            print('Error: Invalid reconstructor.')
            sys.exit()

        if np.isnan(np.sum(L)) or np.isnan(np.sum(R)):
            print("Error, nan in array, function: reconstruction")
            sys.exit()

        return L, R


    def _riemann(self, Q, g, p, axis):
        '''
        Returns the difference in flux dF/dxi 
        between the left and right cell edges
        in the .
        '''

        #Initialise arrays to hold flux vales.
        if axis == 'i':
            flux = np.zeros(shape=g.shape_flux_x1)
            FR = np.zeros(shape=g.shape_flux_x1)
            FL = np.zeros(shape=g.shape_flux_x1)
        if axis == 'j':
            flux = np.zeros(shape=g.shape_flux_x2)
            FR = np.zeros(shape=g.shape_flux_x2)
            FL = np.zeros(shape=g.shape_flux_x2)

        # Obtain the states on the left (L) and right (R) faces of 
        # the cell for the conservative variables.
        QL, QR = self._reconstruction(Q, g, p, axis)

        # Construct the flux tensor 
        # either side of interface.
        FL = self._flux_tensor(QL, axis, p['gamma'], p['Dimensions'])
        FR = self._flux_tensor(QR, axis, p['gamma'], p['Dimensions'])

        if p['riemann'] == 'hll':

            SL, SR = self._eigenvalues(QL, QR, p['gamma'], axis, p['Dimensions'])

            slicing1 = np.where(SL > 0.0)
            slicing2 = np.where((SL <= 0.0) & (0.0 <= SR))
            slicing3 = np.where(SR < 0.0)

            flux[:, slicing1] = FL[:, slicing1]
            flux[:, slicing2] = SR[slicing2]*FL[:, slicing2] \
                                - SL[slicing2]*FR[:, slicing2] \
                                + SL[slicing2]*SR[slicing2] *(QR[:, slicing2] \
                                - QL[:, slicing2])/(SR[slicing2] - SL[slicing2])
            flux[:, slicing3] = FR[:, slicing3]
                
        if np.isnan(np.sum(flux)):
            print("Error, nan in array, function: riemann")
            sys.exit()

        return flux

    def _flux_tensor(self, Q, axis, gamma, dim):

        F = np.zeros(shape=Q.shape)

        U = self._cons_to_prims(Q, gamma, dim)

        if dim == '1D':
            F[rho] = U[rho]*U[vx1]
            F[mvx1] = U[rho]*U[vx1]**2 + U[prs]
            F[eng] = U[vx1]*(Q[eng] + U[prs])

        if axis == 'i' and dim == '2D':
            F[rho] = U[rho]*U[vx1]
            F[mvx1] = U[rho]*U[vx1]**2 + U[prs]
            F[mvx2] = U[rho]*U[vx1]*U[vx1]
            F[eng] = U[vx1]*(Q[eng] + U[prs])
        elif axis == 'j' and dim == '2D':
            F[rho] = U[rho]*U[vx2]
            F[mvx1] = U[rho]*U[vx1]*U[vx2]
            F[mvx2] = U[rho]*U[vx2]**2 + U[prs]
            F[eng] = U[vx2]*(Q[eng] + U[prs])

        return F


    def _eigenvalues(self, QL, QR, gamma, axis, dim):

        UL = self._cons_to_prims(QL, gamma, dim)
        UR = self._cons_to_prims(QR, gamma, dim)

        csL = np.sqrt(gamma*UL[prs]/UL[rho])
        csR = np.sqrt(gamma*UR[prs]/UR[rho])

        if(np.isnan(csL).any() or np.isnan(csR).any()):
            print("Erroar, nan found in eigenvalues:")
            print('csL=', csL)
            print('csR=', csR)
            print('prs=', UL[prs])
            print('rho=', UL[rho])
            sys.exit()

        if axis == 'i':
            Sp = np.maximum(abs(UL[u]) + csL, abs(UR[u]) + csR)
        elif axis == 'j':
            Sp = np.maximum(abs(UL[v]) + csL, abs(UR[v]) + csR)

        SL = -Sp
        SR = Sp

        return SL, SR


    def _cons_to_prims(self, Q, gamma, dim):

        U = np.zeros(shape=Q.shape)

        m2 = Q[mvx1]*Q[mvx1]
        if dim == '2D':
            m2 += Q[mvx2]*Q[mvx2]

        kinE = 0.5*m2/Q[rho]

        U[rho] = Q[rho]
        U[vx1] = Q[mvx1]/Q[rho]
        U[prs] = (gamma - 1.0)*(Q[eng] - kinE)
        
        if dim == '2D':
            U[vx2] = Q[mvx2]/Q[rho]

        if np.isnan(U).any():
            print("Error, nan in cons_to_prims")
            sys.exit()

        return U


    def _prims_to_cons(self, U, gamma, dim):

        Q = np.zeros(shape=U.shape)

        v2 = U[vx1]*U[vx1]
        if dim == '2D':
            v2 += U[vx2]*U[vx2]

        Q[rho] = U[rho]
        Q[mvx1] = U[rho]*U[vx1]
        Q[eng] = 0.5*U[rho]*v2 + U[prs]/(gamma - 1.0)

        if dim == '2D':
            Q[mvx2] = U[rho]*U[vx2]

        if np.isnan(Q).any():
            print("Error, nan in prims_to_cons")
            sys.exit()

        return Q 


    def time_step(self, U, g, gamma, cfl, dim):

        cs = np.sqrt(gamma*U[prs]/U[rho])

        if dim == '1D':
            max_velocity = np.amax(abs(U[vx1]))
            max_speed = np.amax(abs(U[vx1]) + cs)
            dt = cfl*g.dx1/max_speed 
            mach_number = np.amax(abs(U[vx1])/cs)
        elif dim == '2D':
            max_velocity = np.amax(abs(U[vx1:vx2]))
            max_speed = np.amax(abs(U[vx1:vx2]) + cs)
            dt = cfl*min(g.dx1, g.dx2)/max_speed 
            mach_number = np.amax(abs(U[vx1:vx2])/cs)

        if np.isnan(dt):
            print("Error, nan in time_step", cs)
            sys.exit()

        return dt, max_velocity, mach_number




