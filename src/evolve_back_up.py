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

        if para['time stepping'] == 'Euler':
            U_old = U
            U_new = np.zeros_like(U)
            for i in range(g.ibeg(), g.iend()):                                       
                for j in range(g.jbeg(), g.jend()):
                    dFdx, dFdy = self.Riemann(U[:, i-g.nxg:i+g.nxg+1, j-g.nyg:j+g.nyg+1], g, para)
                    for var in range(g.nvar):
                        U_new[var, i, j] = U_old[var, i, j] - dt*(dFdx[var] + dFdy[var])

            
        if para['time stepping'] == 'RK2':
            U_old = U
            U_new = np.zeros_like(U)

            # k1 step.         
            for i in range(g.ibeg(), g.iend()):                                       
                for j in range(g.jbeg(), g.jend()):
                    dFdx, dFdy = self.Riemann(U_old[:, i-g.nxg:i+g.nxg+1, j-g.nyg:j+g.nyg+1], g, para)
                    for var in range(g.nvar):
                        U[var, i, j] = U_old[var, i, j] - 0.5*dt*(dFdx[var] + dFdy[var])

            # Reaply boundary conditions.
            Boundary(U, g, para)

            # k2 step.
            for i in range(g.ibeg(), g.iend()):                                       
                for j in range(g.jbeg(), g.jend()):
                    dFdx, dFdy = self.Riemann(U[:, i-g.nxg:i+g.nxg+1, j-g.nyg:j+g.nyg+1], g, para)
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
           |   |   |   |
         2 |___|___|___|_1 R
        j  |   |   |   | 0
         1 |___|___|___|_1 L
           |   |   |   | 0
         0 |___|___|___|
             0   1   2  
                 i

        Linear:
                   L   R
                  0 1 0 1
            ___ ___|___|___ ___
           |   |   |   |   |   |
         4 |___|___|___|___|___|
           |   |   |   |   |   |
         3 |___|___|___|___|___|_1 R
        j  |   |   |   |   |   | 0
         2 |___|___|___|___|___|_1 L
           |   |   |   |   |   | 0
         1 |___|___|___|___|___|
           |   |   |   |   |   |
         0 |___|___|___|___|___|
             0   1   2   3   4
                     i
        '''
        xL = np.zeros((g.nvar, 2))
        xR = np.zeros_like(xL)
        yL = np.zeros_like(xL)
        yR = np.zeros_like(xL)
        if para['reconstruction'] == 'flat':
            i = 1
            j = 1
            xL[:, 0] = a[:, i-1, j]
            xL[:, 1] = a[:, i, j]
            xR[:, 0] = a[:, i, j]
            xR[:, 1] = a[:, i+1, j]
            yL[:, 0] = a[:, i, j-1]
            yL[:, 1] = a[:, i, j]
            yR[:, 0] = a[:, i, j]
            yR[:, 1] = a[:, i, j+1]

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

            i = 2
            j = 2
            xL[:, 0] = linear(a[:, i-2:i+1, j], g.dx(), g.nvar, para)
            xL[:, 1] = linear(a[:, i-1:i+2, j], g.dx(), g.nvar, para)
            xR[:, 0] = linear(a[:, i-1:i+2, j], g.dx(), g.nvar, para)
            xR[:, 1] = linear(a[:, i:i+3, j], g.dx(), g.nvar, para)
            yL[:, 0] = linear(a[:, i, j-2:j+1], g.dy(), g.nvar, para)
            yL[:, 1] = linear(a[:, i, j-1:j+2], g.dy(), g.nvar, para)
            yR[:, 0] = linear(a[:, i, j-1:j+2], g.dy(), g.nvar, para)
            yR[:, 1] = linear(a[:, i, j:j+3], g.dy(), g.nvar, para) 


        return xL, xR, yL, yR

    def Riemann(self, U, g, para):
        '''
        Returns the difference in flux dF/dxi 
        between the left and right cell edges
        for both the x and y dimensions.
        '''
        rho = 0
        prs = 1
        vx = 2
        vy = 3
        eng = 1

        FLx = np.zeros((g.nvar, 2))
        FRx = np.zeros_like(FLx)
        FLy = np.zeros_like(FLx)
        FRy = np.zeros_like(FLx)

        lower_flux_x = np.zeros(g.nvar)
        upper_flux_x = np.zeros(g.nvar)
        lower_flux_y = np.zeros(g.nvar)
        upper_flux_y = np.zeros(g.nvar)

        q = self.prims_to_cons(U, para)
        qxL, qxR, qyL, qyR = self.Reconstruction(q, g, para)
        xL, xR, yL, yR = self.Reconstruction(U, g, para)

        if para['riemann'] == 'hll':

            for m in range(1):
                # Lower x face:
                FLx[0, m] = xL[rho, m]*xL[vx, m]
                FLx[1, m] = (xL[prs, m]*qxL[eng, m] + 0.5*xL[rho, m]*xL[vx, m]*xL[vx, m] + xL[prs, m])*xL[vx, m]
                FLx[2, m] = xL[rho, m]*xL[vx, m]*xL[vx, m] + xL[prs, m]
                FLx[3, m] = xL[rho, m]*xL[vy, m]*xL[vx, m]

                # Upper x face:
                FRx[0, m] = xR[rho, m]*xR[vx, m]
                FRx[1, m] = (xR[prs, m]*qxR[eng, m] + 0.5*xR[rho, m]*xR[vx, m]*xR[vx, m] + xR[prs, m])*xR[vx, m]
                FRx[2, m] = xR[rho, m]*xR[vx, m]*xR[vx, m] + xR[prs, m]
                FRx[3, m] = xR[rho, m]*xR[vy, m]*xR[vx, m]

                # Lower y face:
                FLy[0, m] = yL[rho, m]*yL[vy, m]
                FLy[1, m] = (yL[prs, m]*qyL[eng, m] + 0.5*yL[rho, m]*yL[vy, m]*yL[vy, m] + yL[prs, m])*yL[vy, m]
                FLy[2, m] = yL[rho, m]*yL[vx, m]*yL[vy, m]
                FLy[3, m] = yL[rho, m]*yL[vy, m]*yL[vy, m] + yL[prs, m]

                # Upper y face:
                FRy[0, m] = yR[rho, m]*yR[vy, m]
                FRy[1, m] = (yR[prs, m]*qyR[eng, m] + 0.5*yR[rho, m]*yR[vy, m]*yR[vy, m] + yR[prs, m])*yR[vy, m]
                FRy[2, m] = yR[rho, m]*yR[vx, m]*yR[vy, m]
                FRy[3, m] = yR[rho, m]*yR[vy, m]*yR[vy, m] + yR[prs, m]

            for var in range(g.nvar):

                # Lower x face:
                lam = self.Eigenvalues(xL[rho, :], xL[prs, :], xL[vx, :], para)
                smax = max(lam)
                smin = min(lam)
                if (smin >= 0.0):
                    lower_flux_x[var] = FLx[var, 0]
                if (smin <= 0.0 <= smax):
                    lower_flux_x[var] = smax*FLx[var, 0] - smin*FLx[var, 1] + smin*smax*(qxL[var, 1] - qxL[var, 0])/(smax-smin)
                if (smax <= 0.0):
                    lower_flux_x[var] = FLx[var, 1]
                
                # Upper x face:
                lam = self.Eigenvalues(xR[rho, :], xR[prs, :], xR[vx, :], para)
                smax = max(lam)
                smin = min(lam)
                if (smin >= 0.0):
                    upper_flux_x[var] = FRx[var, 0] 
                if (smin <= 0.0 <= smax):
                    upper_flux_x[var] = smax*FRx[var, 0] - smin*FRx[var, 1] + smin*smax*(qxR[var, 1] - qxR[var, 0])/(smax-smin) 
                if (smax <= 0.0):
                    upper_flux_x[var] = FRx[var, 1]

                # Lower y face:
                lam = self.Eigenvalues(yL[rho, :], yL[prs, :], yL[vy, :], para)
                smax = max(lam)
                smin = min(lam)
                if (smin >= 0.0):
                    lower_flux_y[var] = FLy[var, 0]
                if (smin <= 0.0 <= smax):
                    lower_flux_y[var] = smax*FLy[var, 0] - smin*FLy[var, 1] + smin*smax*(qyL[var, 1] - qyL[var, 0])/(smax-smin)
                if (smax <= 0.0):
                    lower_flux_y[var] = FLy[var, 1]

                # Upper y face
                lam = self.Eigenvalues(yR[rho, :], yR[prs, :], yR[vy, :], para)
                smax = max(lam)
                smin = min(lam)
                if (smin >= 0.0):
                    upper_flux_y[var] = FRy[var, 0]
                if (smin <= 0.0 <= smax):
                    upper_flux_y[var] = smax*FRy[var, 0] - smin*FRy[var, 1] + smin*smax*(qyR[var, 1] - qyR[var, 0])/(smax-smin) 
                if (smax <= 0.0):
                    upper_flux_y[var] = FRy[var, 1]

            dFdx = (upper_flux_x - lower_flux_x)/g.dx()
            dFdy = (upper_flux_y - lower_flux_y)/g.dy()

        return dFdx, dFdy

    def Eigenvalues(self, rho, prs, v, para):
        csL = para['gamma']*prs[0]/rho[0]
        csR = para['gamma']*prs[1]/rho[1]
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
        U[rho, :, :] = q[rho,:,:]
        U[prs, :, :] = q[rho,:,:]*q[eng, :, :]*(gamma - 1.0)
        U[vx1, :, :] = q[mvx2,:,:]/U[rho,:,:]
        U[vx2, :, :] = q[mvx2,:,:]/U[rho,:,:]
        U[np.isnan(U[:, :, :])] = 0.0
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
        q[rho, :, :] = U[rho,:,:]
        q[eng, :, :] = U[prs,:,:]/(U[rho,:,:]*(gamma - 1.0))
        q[mvx1, :, :] = U[rho,:,:]*U[vx1,:,:]
        q[mvx2, :, :] = U[rho,:,:]*U[vx2,:,:]
        q[np.isnan(q[:, :, :])] = 0.0
        return q       

    def time_step(self, U, g, para):
        rho = 0
        prs = 1
        vx1 = 2
        vx2 = 3
        cs = para['gamma']*U[prs, g.ibeg():g.iend(), g.jbeg():g.jend()] \
                          /U[rho, g.ibeg():g.iend(), g.jbeg():g.jend()]
        a = np.amax(abs(U[vx1, g.ibeg():g.iend(), g.jbeg():g.jend()]))#-cs)
        b = np.amax(abs(U[vx2, g.ibeg():g.iend(), g.jbeg():g.jend()]))#-cs)
        dt = para['cfl']*min(g.dx(), g.dy())/max(a, b) 
        return dt, a, b
