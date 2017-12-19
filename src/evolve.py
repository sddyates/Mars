import numpy as np
from globe import *
import sys
from output import mesh_plot, line_plot
import matplotlib.pyplot as plt

class Evolve:

    def __init__(self):
        None

    def incriment(self, V, dt, g, p):
        """
        Synopsis
        --------
        Evolve the simulation domain though one 
        time step.

        Args
        ----
        V: numpy array-like
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
        U = self._prims_to_cons(V, p['gamma'], p['Dimensions'])

        # Time step with Euler.
        if p['time stepping'] == 'Euler':
            U_new = U + dt*self._RHSOperator(U, g, p)

        
        # Time step with RK2.
        elif p['time stepping'] == 'RK2':

            # k1 step.
            U_star = U + 0.5*dt*self._RHSOperator(U, g, p)

            # Reaply boundary conditions.
            g.boundary(U_star, p)

            # k2 step.
            U_new = 0.5*(U + U_star + dt*self._RHSOperator(U_star, g, p))
        
        else:
            print('Error: Invalid integrator.')
            sys.exit()

        # Convert primitive to conservative variables.
        V = self._cons_to_prims(U_new, p['gamma'], p['Dimensions'])

        if np.isnan(np.sum(V)):
            print("Error, nan in array, function: incriment")
            sys.exit()

        return V

    def _RHSOperator(self, U, g, p):

        dflux_x1 = np.zeros(shape=U.shape)

        if p['Dimensions'] == '1D':
            face_fluxes, press = self._riemann(U, g, p, 'i')
            Fneg = face_fluxes[:, :-1]
            Fpos = face_fluxes[:, 1:]

            dflux_x1[:, g.ibeg:g.iend] = -(Fpos - Fneg)
            #dflux_x1[mvx1, g.ibeg:g.iend] -= (press[1:] - press[:-1])
            dflux = dflux_x1/g.dx1

        if p['Dimensions'] == '2D':
            dflux_x2 = np.zeros(shape=U.shape)
            # Transvers loop over columns in x1.
            for j in range(g.jbeg, g.jend):
                face_flux_x1 = self._riemann(U[:, j, :], g, p, 'i')
                Fneg = face_flux_x1[:, :-1]
                Fpos = face_flux_x1[:, 1:]
                dflux_x1[:, j, g.ibeg:g.iend] = -(Fpos - Fneg)/g.dx1

            # Transvers loop over columns in x2.
            for i in range(g.ibeg, g.iend):                                       
                face_flux_x2 = self._riemann(U[:, :, i], g, p, 'j')
                Fneg = face_flux_x2[:, :-1]
                Fpos = face_flux_x2[:, 1:]
                dflux_x2[:, g.jbeg:g.jend, i] = -(Fpos - Fneg)/g.dx2

            dflux = dflux_x1 + dflux_x2

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
            L = y[:, :-g.gz]
            R = y[:, g.gz:]

        elif p['reconstruction'] == 'linear':

            def linear(y, dxi, p, side):                

                if p['limiter'] == None:                    
                    m = np.gradient(y, dxi, axis=1)
                    m = m[:, g.gz - 1:-g.gz + 1]

                elif p['limiter'] == 'minmod':               

                    m = np.zeros(shape=y[:, 1:-1].shape)
                    for i in range(1, len(y)):

                        for var in range(g.nvar):
                            a = (y[var, i] - y[var, i-1])/dxi
                            b = (y[var, i+1] - y[var, i])/dxi
    
                            if (abs(a) < abs(b)) & (a*b > 0.0):
                                m[var, i] = a
                            elif (abs(a) > abs(b)) & (a*b > 0.0):
                                m[var, i] = b
                            elif a*b <= 0.0:
                                m[var, i] = 0.0

                    '''
                    a = (y[:, 1:-1] - y[:, :-g.gz])/dxi 
                    b = (y[:, g.gz:] - y[:, 1:-1])/dxi
                    m = np.zeros(shape=a.shape)

                    slicing1 = np.where((abs(a) < abs(b)) & (a*b > 0.0))
                    slicing2 = np.where((abs(a) > abs(b)) & (a*b > 0.0))
                    slicing3 = np.where(a*b <= 0.0)
                    '''
                    #print(m)
                    #m[slicing1] = a[slicing1]
                    #m[slicing2] = b[slicing2]
                    #m[slicing3] = 0.0
                    #print('post',m)

                else:
                    print('Error: Invalid limiter.')
                    sys.exit()

                if side == 'left':
                    return y[:, g.gz - 1:-g.gz] \
                           + m[:, :-1]/2.0*dxi
                if side == 'right':
                    return y[:, g.gz:-g.gz + 1] \
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


    def _riemann(self, U, g, p, axis):
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
            press = np.zeros(shape=g.shape_flux_x1[1])
        if axis == 'j':
            flux = np.zeros(shape=g.shape_flux_x2)
            FR = np.zeros(shape=g.shape_flux_x2)
            FL = np.zeros(shape=g.shape_flux_x2)
            press = np.zeros(shape=g.shape_flux_x2[2])

        # Obtain the states on the left (L) and right (R) faces of 
        # the cell for the conservative variables.
        UL, UR = self._reconstruction(U, g, p, axis)
        VL = self._cons_to_prims(UL, p['gamma'], p['Dimensions'])
        VR = self._cons_to_prims(UR, p['gamma'], p['Dimensions'])

        # Construct the flux tensor 
        # either side of interface.
        FL = self._flux_tensor(UL, axis, p['gamma'], p['Dimensions'])
        FR = self._flux_tensor(UR, axis, p['gamma'], p['Dimensions'])

        if p['riemann'] == 'tvdlf':
            Smax = abs(np.amax(self._eigenvalues(UL, UR, p['gamma'], 
                               axis, p['Dimensions'])))
            flux = 0.5*(FL + FR - Smax*(UR - UL))
            press = 0.0

        if p['riemann'] == 'hll':

            SL, SR = self._eigenvalues(UL, UR, p['gamma'], axis, p['Dimensions'])

            for var in range(g.nvar):
                for i in range(len(flux[0, :])):

                    if SL[i] > 0.0:
                        flux[var, i] = FL[var, i]
                        press[i] = VL[prs, i]
                    elif (SR[i] <= 0.0):
                        flux[var, i] = FR[var, i]
                        press[i] = VR[prs, i]
                    else:
                        flux[var, i] = (SR[i]*FL[var, i] \
                                       - SL[i]*FR[var, i] \
                                       + SL[i]*SR[i]*(UR[var, i] \
                                      - UL[var, i]))/(SR[i] - SL[i])
                        scrh = 1.0/(SR[i] - SL[i])
                        press[i] = (SR[i]*VL[prs, i] - SL[i]*VR[prs, i])*scrh
            '''
            slicing1 = np.where(SL > 0.0)
            slicing2 = np.where((SL <= 0.0) & (0.0 <= SR))
            slicing3 = np.where(SR < 0.0)

            flux[:, slicing1] = FL[:, slicing1]
            flux[:, slicing2] = SR[slicing2]*FL[:, slicing2] \
                                - SL[slicing2]*FR[:, slicing2] \
                                + SL[slicing2]*SR[slicing2] *(UR[:, slicing2] \
                                - UL[:, slicing2])/(SR[slicing2] - SL[slicing2])
            flux[:, slicing3] = FR[:, slicing3]
            '''
 
        if np.isnan(np.sum(flux)):
            print("Error, nan in array, function: riemann")
            sys.exit()

        return flux, press

    def _flux_tensor(self, U, axis, gamma, dim):

        F = np.zeros(shape=U.shape)

        V = self._cons_to_prims(U, gamma, dim)

        if dim == '1D':
            F[rho] = V[rho]*V[vx1]
            F[mvx1] = V[rho]*V[vx1]**2 + V[prs]
            F[eng] = V[vx1]*(U[eng] + V[prs])

        if axis == 'i' and dim == '2D':
            F[rho] = V[rho]*V[vx1]
            F[mvx1] = V[rho]*V[vx1]**2 + V[prs]
            F[mvx2] = V[rho]*V[vx1]*V[vx2]
            F[eng] = V[vx1]*(U[eng] + V[prs])
        elif axis == 'j' and dim == '2D':
            F[rho] = V[rho]*V[vx2]
            F[mvx1] = V[rho]*V[vx1]*V[vx2]
            F[mvx2] = V[rho]*V[vx2]**2 + V[prs]
            F[eng] = V[vx2]*(U[eng] + V[prs])

        return F


    def _eigenvalues(self, UL, UR, gamma, axis, dim):

        VL = self._cons_to_prims(UL, gamma, dim)
        VR = self._cons_to_prims(UR, gamma, dim)

        csL = np.sqrt(gamma*VL[prs]/VL[rho])
        csR = np.sqrt(gamma*VR[prs]/VR[rho])

        if(np.isnan(csL).any() or np.isnan(csR).any()):
            print("Erroar, nan found in eigenvalues:")
            print('csL=', csL)
            print('csR=', csR)
            print('VL[rho]=', VL[rho])
            print('VL[prs]=', VL[prs])
            print('VR[rho]=', VR[rho])
            print('VR[prs]=', VR[prs])
            sys.exit()

        if axis == 'i':
            Sp = np.maximum(abs(VL[u]) + csL, abs(VR[u]) + csR)
        elif axis == 'j':
            Sp = np.maximum(abs(VL[v]) + csL, abs(VR[v]) + csR)

        SL = -Sp
        SR = Sp

        return SL, SR


    def _cons_to_prims(self, U, gamma, dim):

        V = np.zeros(shape=U.shape)

        m2 = U[mvx1]*U[mvx1]
        if dim == '2D':
            m2 += U[mvx2]*U[mvx2]

        kinE = 0.5*m2/U[rho]

        V[rho] = U[rho]
        V[vx1] = U[mvx1]/U[rho]
        V[prs] = (gamma - 1.0)*(U[eng] - kinE)
        
        if dim == '2D':
            V[vx2] = U[mvx2]/U[rho]

        if np.isnan(V).any():
            print("Error, nan in cons_to_prims")
            sys.exit()

        return V


    def _prims_to_cons(self, V, gamma, dim):

        U = np.zeros(shape=V.shape)

        v2 = V[vx1]*V[vx1]
        if dim == '2D':
            v2 += V[vx2]*V[vx2]

        U[rho] = V[rho]
        U[mvx1] = V[rho]*V[vx1]
        U[eng] = 0.5*V[rho]*v2 + V[prs]/(gamma - 1.0)

        if dim == '2D':
            U[mvx2] = V[rho]*V[vx2]

        if np.isnan(U).any():
            print("Error, nan in prims_to_cons")
            sys.exit()

        return U 


    def time_step(self, V, g, gamma, cfl, dim):

        cs = np.sqrt(gamma*V[prs]/V[rho])

        if dim == '1D':
            max_velocity = np.amax(abs(V[vx1]))
            max_speed = np.amax(abs(V[vx1]) + cs)
            dt = cfl*g.dx1/max_speed 
            mach_number = np.amax(abs(V[vx1])/cs)
        elif dim == '2D':
            max_velocity = np.amax(abs(V[vx1:vx2]))
            max_speed = np.amax(abs(V[vx1:vx2]) + cs)
            dt = cfl*min(g.dx1, g.dx2)/max_speed 
            mach_number = np.amax(abs(V[vx1:vx2])/cs)

        if np.isnan(dt):
            print("Error, nan in time_step", cs)
            sys.exit()

        return dt, max_velocity, mach_number




