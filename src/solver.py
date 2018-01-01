import numpy as np
from globe import *
from tools import cons_to_prims, eigenvalues

def riennman(s, g, p, axis):

    def tvdlf(s):
        Smax = max(np.amax(abs(s.SL)), np.amax(abs(s.SR)))
        return 0.5*(s.FL + s.FR - Smax*(s.UR - s.UL))


    def hll(s, g):
        for var in range(g.nvar):
            for i in range(len(s.flux[0, :])):
                if s.SL[i] > 0.0:
                    s.flux[var, i] = s.FL[var, i]
                elif (SR[i] < 0.0):
                    s.flux[var, i] = s.FR[var, i]
                else:
                    s.flux[var, i] = (s.SR[i]*s.FL[var, i] \
                                   - s.SL[i]*s.FR[var, i] \
                                   + s.SL[i]*s.SR[i]*(s.UR[var, i] \
                                   - s.UL[var, i]))/(s.SR[i] - s.SL[i])
                    scrh = 1.0/(s.SR[i] - s.SL[i])
        return


    def hllc(s, g, p, axis):

        '''
        VL = cons_to_prims(s.UL, p)
        VR = cons_to_prims(s.UR, p)

        if p['Dimensions'] == '1D':
            uL = s.VL[vx1]
            uR = s.VR[vx1]
            vL = 0.0
            vR = 0.0
        elif p['Dimensions'] == '2D':
            uL = s.VL[vx1] if axis == 'i' else s.VL[vx2]
            vL = s.VL[vx2] if axis == 'i' else s.VL[vx1]
            uR = s.VR[vx1] if axis == 'j' else s.VR[vx2]
            vR = s.VR[vx2] if axis == 'j' else s.VR[vx1]

        S = (s.VR[prs] - s.VL[prs] \
           + s.VL[rho]*uL*(s.SL - uL) \
           - s.VR[rho]*uR*(s.SR - uR))\
           /(s.VL[rho]*(s.SL - uL) - s.VR[rho]*(s.SR - uR))

        def hllc_U_star(UK, VK, SK, uK, vK, S):

            U_star_K = VK[rho]*(SK - uK)/(SK - S)

            E_factor = UK[eng]/VK[rho] + (S - uK)\
                *(S + VK[prs]/(VK[rho]*(SK - uK)))

            vec = np.array([np.ones(S.shape), E_factor, S])

            return U_star_K*vec

        U_star_L = hllc_U_star(s.UL, s.VL, s.SL, uL, vL, S)
        U_star_R = hllc_U_star(s.UR, s.VR, s.SR, uR, vR, S) 

        F_star_L = s.FL + s.SL*(U_star_L - s.UL)
        F_star_R = s.FR + s.SR*(U_star_R - s.UR)

        for var in range(g.nvar):
            for i in range(len(s.flux[0, :])):

                if s.SL[i] > 0.0:
                    s.flux[var, i] = s.FL[var, i]
                elif (s.SL[i] < 0.0 < S[i]):
                    s.flux[var, i] = F_star_L[var, i]
                elif (S[i] < 0.0 < s.SR[i]):
                    s.flux[var, i] = F_star_R[var, i]
                elif (s.SR[i] < 0.0):
                    s.flux[var, i] = s.FR[var, i]
        '''
        if p['Dimensions'] == '1D':
            mxn = mvx1
            vxn = vx1
        elif p['Dimensions'] == '2D':
            mxn = mvx1 if axis == 'i' else mvx2
            mxt = mvx2 if axis == 'i' else mvx1
            vxn = vx1 if axis == 'i' else vx2
            vxt = vx2 if axis == 'i' else vx1

        for i in range(len(s.flux[0, :])):

            if s.SL[i] > 0.0:
                s.flux = s.FL

            elif s.SR[i] < 0.0:
                s.flux = s.FR

            else:

                USL = np.zeros(shape=s.UL.shape[0])
                USR = np.zeros(shape=s.UR.shape[0])

                QL = s.VL[prs, i] + s.UL[mxn, i]*(s.VL[vxn, i] - s.SL[i])
                QR = s.VR[prs, i] + s.UR[mxn, i]*(s.VR[vxn, i] - s.SR[i])

                WL = s.VL[rho, i]*(s.VL[vxn, i] - s.SL[i])
                WR = s.VR[rho, i]*(s.VR[vxn, i] - s.SR[i])

                VS = (QR - QL)/(WR - WL)

                USL[rho] = s.UL[rho, i]*(s.SL[i] - s.VL[vxn, i])/(s.SL[i] - VS)
                USR[rho] = s.UR[rho, i]*(s.SR[i] - s.VR[vxn, i])/(s.SR[i] - VS)

                USL[mxn] = USL[rho]*VS
                USR[mxn] = USR[rho]*VS
                if p['Dimensions'] == '2D':
                    USL[mxt] = USL[rho]*s.VL[vxt, i]
                    USR[mxt] = USR[rho]*s.VR[vxt, i]

                USL[eng] = s.UL[eng, i]/s.VL[rho, i] \
                           + (VS - s.VL[vxn, i])*(VS + s.VL[prs, i]/(s.VL[rho, i]*(s.SL[i] - s.VL[vxn, i])));
                USR[eng] = s.UR[eng, i]/s.VR[rho, i] \
                           + (VS - s.VR[vxn, i])*(VS + s.VR[prs, i]/(s.VR[rho, i]*(s.SR[i] - s.VR[vxn, i])));

                USL[eng] *= USL[rho];
                USR[eng] *= USR[rho];

                if VS >= 0.0:
                    s.flux[:, i] = s.FL[:, i] + s.SL[i]*(USL - s.UL[:, i]);
                else:
                    s.flux[:, i] = s.FR[:, i] + s.SR[i]*(USR - s.UR[:, i]);


        return
 

    if p['riemann'] == 'tvdlf':
        tvdlf(s)
    elif p['riemann'] == 'hll':
        hll(s, g)
    elif p['riemann'] == 'hllc':
        hllc(s, g, p, axis)
    else:
        print('Error: invalid riennman solver.')
        sys.exit()

    if np.isnan(np.sum(s.flux)):
        print("Error, nan in array, function: riemann")
        sys.exit()

    return


