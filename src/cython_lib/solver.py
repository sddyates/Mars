import numpy as np
from globe import *
from tools import cons_to_prims, eigenvalues


def tvdlf(s):
    Smax = max(np.amax(abs(s.SL)), np.amax(abs(s.SR)))
    return 0.5*(s.FL + s.FR - Smax*(s.UR - s.UL))


def hll(s, g):
    for var in range(g.nvar):
        for i in range(len(s.flux[0, :])):
            if s.SL[i] > 0.0:
                s.flux[var, i] = s.FL[var, i]
            elif (s.SR[i] < 0.0):
                s.flux[var, i] = s.FR[var, i]
            else:
                s.flux[var, i] = (s.SR[i]*s.FL[var, i] \
                               - s.SL[i]*s.FR[var, i] \
                               + s.SL[i]*s.SR[i]*(s.UR[var, i] \
                               - s.UL[var, i]))/(s.SR[i] - s.SL[i])
                scrh = 1.0/(s.SR[i] - s.SL[i])
    return


def hllc(s, g, p, axis):

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
 



