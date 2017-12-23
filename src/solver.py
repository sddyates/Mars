import numpy as np
from globe import *
from tools import cons_to_prims, eigenvalues

def riennman(flux, FL, FR, SL, SR, UL, UR, 
             VL, VR, g, axis, gamma, dim, solver):

    def tvdlf(flux, FL, FR, SL, SR):
        Smax = abs(np.amax(SL, SR))
        flux = 0.5*(FL + FR - Smax*(UR - UL))
        return


    def hll(flux, FL, FR, SL, SR, UL, UR, g):
        for var in range(g.nvar):
            for i in range(len(flux[0, :])):
                if SL[i] > 0.0:
                    flux[var, i] = FL[var, i]
                elif (SR[i] < 0.0):
                    flux[var, i] = FR[var, i]
                else:
                    flux[var, i] = (SR[i]*FL[var, i] \
                                   - SL[i]*FR[var, i] \
                                   + SL[i]*SR[i]*(UR[var, i] \
                                   - UL[var, i]))/(SR[i] - SL[i])
                    scrh = 1.0/(SR[i] - SL[i])
        return


    def hllc(flux, FL, FR, SL, SR, UL, UR, VL, 
              VR, g, axisi, gamma, dim):

        VL = cons_to_prims(UL, gamma, dim)
        VR = cons_to_prims(UR, gamma, dim)

        uL = VL[vx1]# if axis == 'i' else VL[vx2]
        vL = 0.0#VL[vx2] if axis == 'i' else VL[vx1]

        uR = VR[vx1]# if axis == 'j' else VR[vx2]
        vR = 0.0#VR[vx2] if axis == 'j' else VR[vx1]

        S = (VR[prs] - VL[prs] \
           + VL[rho]*uL*(SL - uL) \
           - VR[rho]*uR*(SR - uR))\
           /(VL[rho]*(SL - uL) - VR[rho]*(SR - uR))

        def hllc_U_star(UK, VK, SK, uK, vK, S):
            UK_star = VK[rho]*(SK - uK)/(SK - S)
            E_factor = UK[eng]/VK[rho] + (S - uK)\
                *(S + VK[prs]/(VK[rho]*(SK - uK)))
            vec = np.array([np.ones(S.shape), E_factor, S])
            return UK_star*vec

        UL_star = hllc_U_star(UL, VL, SL, uL, vL, S)
        UR_star = hllc_U_star(UR, VR, SR, uR, vR, S) 

        FL_star = FL + SL*(UL_star - UL)
        FR_star = FR + SR*(UR_star - UR)

        for var in range(g.nvar):
            for i in range(len(flux[0, :])):

                if SL[i] > 0.0:
                    flux[var, i] = FL[var, i]
                elif (SR[i] < 0.0):
                    flux[var, i] = FR[var, i]
                elif (SL[i] < 0.0 < S[i]):
                    flux[var, i] = FL_star[var, i]
                elif (S[i] < 0.0 < SR[i]):
                    flux[var, i] = FR_star[var, i]
        return
 

    if solver == 'tvdlf':
        tvdlf(flux, FL, FR, SL, SR)
    elif solver == 'hll':
        hll(flux, FL, FR, SL, SR, UL, UR, g)
    elif solver == 'hllc':
        hllc(flux, FL, FR, SL, SR, UL, UR, 
                   VL, VR, g, axis, 
                   gamma, Dimensions)
    else:
        print('Error: invalid riennman solver.')
        sys.exit()

    if np.isnan(np.sum(flux)):
        print("Error, nan in array, function: riemann")
        sys.exit()

    return


