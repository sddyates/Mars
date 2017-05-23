import numpy as np
from output import Mesh_plot
from grid import Boundary 

class Evolve:

    def __call__(self, U, g, para):
        t = 0.0
        while (t < para['t_max']):

            dt, vx_max, vy_max = self.time_step(U, g, para)
            U = self.TimeStep(U, dt, g, para)
            t += dt
            Boundary(U, g, para)
            if np.mod(int(t), 1) == 0:
                Mesh_plot(U, g, t)
            print('t = {:.5f}, dt = {:.5f}, vx_max = {:.5f}, vy_max = {:.5f}'.format(t, dt, vx_max, vy_max))

    def TimeStep(self, U, dt, g, para):

        rho = 0
        prs = 1
        vx = 2
        vy =3

        if para['time stepping'] == 'Euler':
            U_old = U
            U_new = np.zeros_like(U)
            for i in range(g.ibeg(), g.iend()-1):                                       
                for j in range(g.jbeg(), g.jend()-1):
                    dFdx = self.Riemann(U[[rho, prs, vx, vy], i-g.nxg:i+g.nxg+1, j], g, para)
                    dFdy = self.Riemann(U[[rho, prs, vy, vx], i, j-g.nyg:j+g.nyg+1], g, para)
                    for var in range(g.nvar):
                        U_new[var, i, j] = U_old[var, i, j] - dt*(dFdx[var] + dFdy[var])

            
        if para['time stepping'] == 'RK2':
            U_old = U
            U_new = np.zeros_like(U)

            # k1 step.         
            for i in range(g.ibeg(), g.iend()-1):                                       
                for j in range(g.jbeg(), g.jend()-1):
                    dFdx = self.Riemann(U[[0, 1, 2, 3], i-g.nxg:i+g.nxg+1, j], g, para)
                    dFdy = self.Riemann(U[[0, 1, 3, 2], i, j-g.nyg:j+g.nyg+1], g, para)
                    for var in range(g.nvar):
                        U[var, i, j] = U_old[var, i, j] - 0.5*dt*(dFdx[var] + dFdy[var])

            # Reaply boundary conditions.
            Boundary(U, g, para)

            # k2 step.
            for i in range(g.ibeg(), g.iend()-1):                                       
                for j in range(g.jbeg(), g.jend()-1):
                    dFdx = self.Riemann(U[[0, 1, 2, 3], i-g.nxg:i+g.nxg+1, j], g, para)
                    dFdy = self.Riemann(U[[0, 1, 3, 2], i, j-g.nyg:j+g.nyg+1], g, para)
                    for var in range(g.nvar):
                        U_new[var, i, j] = U_old[var, i, j] - 0.5*dt*(dFdx[var] + dFdy[var])

        if para['time stepping'] == 'ChrT':
            q = self.prims_to_cons(U, para)
            del(U)

            # Do space loop and Reimann.

            U = self.cons_to_prims(q, para)
            del(q)

        return U_new

    def Reconstruction(self, a, g, para):
        '''
        flat:
            L   R
           0 1 0 1
         ___|___|___
        |   |   |   | 0
        |___|___|___|
          0   1   2  
              i

        Linear:
                L   R
               0 1 0 1
         ___ ___|___|___ ___
        |   |   |   |   |   |
        |___|___|___|___|___|
          0   1   2   3   4
                  i
        '''
        L = np.zeros((g.nvar, 2))
        R = np.zeros_like(L)
        if para['reconstruction'] == 'flat':
            L[:, 0] = a[:, 0]
            L[:, 1] = a[:, 1]
            R[:, 0] = a[:, 1]
            R[:, 1] = a[:, 2]

        if para['reconstruction'] == 'linear':

            def linear(a, dx, nvar, para):
                
                if para['limiter'] == None:
                    grad = (a[:, 2] - a[:, 0])/(2.0*dx)

                if para['limiter'] == 'minmod':
                    grad = np.zeros(nvar)
                    for var in range(nvar):
                        b = (a[var, 1] - a[var, 0])/dx
                        c = (a[var, 2] - a[var, 1])/dx
                        if abs(b) < abs(c) and b*c > 0.0:
                            grad[var] = b
                        elif abs(b) > abs(c) and b*c > 0.0:
                            grad[var] = c
                        else:
                            grad[var] = 0.0
                    
                return a[:, 1] + grad*dx/2.0 

            ''' These are the states on either side of the left 
                and right hand interfaces of the cell. '''
            L[:, 0] = linear(a[:, 0:3], g.dx(), g.nvar, para)
            L[:, 1] = linear(a[:, 1:4], g.dx(), g.nvar, para)
            R[:, 0] = linear(a[:, 1:4], g.dx(), g.nvar, para)
            R[:, 1] = linear(a[:, 2:5], g.dx(), g.nvar, para)

        return L, R

    def Riemann(self, U, g, para):
        '''
        Returns the difference in flux dF/dxi 
        between the left and right cell edges
        for both the x and y dimensions.
        '''
        rho = 0
        prs = 1
        vx1 = 2
        vx2 = 3

        rho = 0
        eng = 1
        mvx1 = 2
        mvx2 = 3

        #U = np.array([RHO, PRS, VX1, VX2])

        FL = np.zeros((g.nvar, 2))
        FR = np.zeros_like(FL)

        lower_flux = np.zeros(g.nvar)
        upper_flux = np.zeros(g.nvar)

        q = self.prims_to_cons(U, para)
        qL, qR = self.Reconstruction(q, g, para)
        UL, UR = self.Reconstruction(U, g, para)

        if para['riemann'] == 'hll':

            for m in range(2):
                # Left of face:
                FL[rho, m] = UL[rho, m]*UL[vx1, m]
                FL[eng, m] = UL[vx1, m]*(qL[eng, m] + UL[prs, m]) #(UL[prs, m]*qL[eng, m] + 0.5*UL[rho, m]*UL[vx1, m]*UL[vx1, m] + UL[prs, m])*UL[vx1, m]
                FL[mvx1, m] = UL[rho, m]*UL[vx1, m]*UL[vx1, m] + UL[prs, m]
                FL[mvx2, m] = UL[rho, m]*UL[vx2, m]*UL[vx1, m]

                # Right of face:
                FR[rho, m] = UR[rho, m]*UR[vx1, m]
                FR[eng, m] = UR[vx1, m]*(qR[eng, m] + UR[prs, m]) #(UR[prs, m]*qR[eng, m] + 0.5*UR[rho, m]*UR[vx1, m]*UR[vx1, m] + UR[prs, m])*UR[vx1, m]
                FR[mvx1, m] = UR[rho, m]*UR[vx1, m]*UR[vx1, m] + UR[prs, m]
                FR[mvx2, m] = UR[rho, m]*UR[vx2, m]*UR[vx1, m]

            for var in range(g.nvar):

                # Lower face:
                UFL = self.cons_to_prims(FL, para)
                lam = self.Eigenvalues(UFL[rho, :], UFL[prs, :], UFL[vx1, :], para)
                smax = max(lam)
                smin = min(lam)
                if (smin >= 0.0):
                    lower_flux[var] = FL[var, 0]
                if (smin <= 0.0 <= smax):
                    lower_flux[var] = smax*FL[var, 0] - smin*FL[var, 1] + smin*smax*(qL[var, 1] - qL[var, 0])/(smax-smin)
                if (smax <= 0.0):
                    lower_flux[var] = FL[var, 1]
                
                # Upper face:
                UFR = self.cons_to_prims(FR, para)
                lam = self.Eigenvalues(UFR[rho, :], UFR[prs, :], UFR[vx1, :], para)
                smax = max(lam)
                smin = min(lam)
                if (smin >= 0.0):
                    upper_flux[var] = FR[var, 0] 
                if (smin <= 0.0 <= smax):
                    upper_flux[var] = smax*FR[var, 0] - smin*FR[var, 1] + smin*smax*(qR[var, 1] - qR[var, 0])/(smax-smin) 
                if (smax <= 0.0):
                    upper_flux[var] = FR[var, 1]
  
            lower_flux = self.cons_to_prims(lower_flux, para)
            upper_flux = self.cons_to_prims(upper_flux, para)
        
            dFdx = (upper_flux - lower_flux)/g.dx()

        return dFdx

    def Eigenvalues(self, rho, prs, v, para):
        csL = np.sqrt(para['gamma']*prs[0]/rho[0])
        csR = np.sqrt(para['gamma']*prs[1]/rho[1])
        if(np.isnan(csL) or np.isnan(csR)):
            print(csL, csR, rho, prs, v)
        lam = [v[0] - csL, 
               v[0], 
               v[0], 
               v[0] + csL, 
               v[1] - csR, 
               v[1], 
               v[1], 
               v[1] + csR] 
        return lam

    def cons_to_prims(self, q, para):
        rho = 0
        prs = 1
        vx1 = 2
        vx2 = 3
        eng = 1
        mvx1 = 2
        mvx2 = 3
        gamma = para['gamma']
        U = np.zeros_like(q)
        U[rho] = q[rho]
        U[prs] = q[rho]*q[eng]*(gamma - 1.0)
        U[vx1] = q[mvx1]/U[rho]
        U[vx2] = q[mvx2]/U[rho]
        U[np.isnan(U[:])] = 0.0
        return U

    def prims_to_cons(self, U, para):
        rho = 0
        prs = 1
        vx1 = 2
        vx2 = 3
        eng = 1
        mvx1 = 2
        mvx2 = 3
        gamma = para['gamma']
        q = np.zeros_like(U)
        q[rho, :] = U[rho,:]
        q[eng, :] = U[prs,:]/(U[rho,:]*(gamma - 1.0))
        q[mvx1, :] = U[rho,:]*U[vx1,:]
        q[mvx2, :] = U[rho,:]*U[vx2,:]
        q[np.isnan(q[:, :])] = 0.0
        return q       

    def time_step(self, U, g, para):
        rho = 0
        prs = 1
        vx1 = 2
        vx2 = 3
        cs = np.sqrt(para['gamma']*U[prs, g.ibeg():g.iend(), g.jbeg():g.jend()] \
                          /U[rho, g.ibeg():g.iend(), g.jbeg():g.jend()])
        a = np.amax(abs(U[vx1, g.ibeg():g.iend(), g.jbeg():g.jend()]) + cs)
        b = np.amax(abs(U[vx2, g.ibeg():g.iend(), g.jbeg():g.jend()]) + cs)
        dt = para['cfl']*min(g.dx(), g.dy())/max(a, b) 
        return dt, a, b
