import numpy as np

def tvdlf(g):
    Smax = max(np.amax(abs(g.SL)), np.amax(abs(g.SR)))
    return 0.5*(g.FL + g.FR - Smax*(g.UR - g.UL))


def hll(g):
    for var in range(g.nvar):
        for i in range(len(g.flux[0, :])):
            if g.SL[i] > 0.0:
                g.flux[var, i] = g.FL[var, i]
            elif (g.SR[i] < 0.0):
                g.flux[var, i] = g.FR[var, i]
            else:
                g.flux[var, i] = (g.SR[i]*g.FL[var, i] \
                               - g.SL[i]*g.FR[var, i] \
                               + g.SL[i]*g.SR[i]*(g.UR[var, i] \
                               - g.UL[var, i]))/(g.SR[i] - g.SL[i])
                scrh = 1.0/(g.SR[i] - g.SL[i])
    return


def hllc(g, p, axis):

    if p['Dimensions'] == '1D':
        mxn = mvx1
        vxn = vx1
    elif p['Dimensions'] == '2D':
        mxn = mvx1 if axis == 'i' else mvx2
        mxt = mvx2 if axis == 'i' else mvx1
        vxn = vx1 if axis == 'i' else vx2
        vxt = vx2 if axis == 'i' else vx1

    for i in range(len(g.flux[0, :])):

        if g.SL[i] > 0.0:
            g.flux = g.FL

        elif g.SR[i] < 0.0:
            g.flux = g.FR

        else:
            USL = np.zeros(shape=g.UL.shape[0])
            USR = np.zeros(shape=g.UR.shape[0])

            QL = g.VL[prs, i] + g.UL[mxn, i]*(g.VL[vxn, i] - g.SL[i])
            QR = g.VR[prs, i] + g.UR[mxn, i]*(g.VR[vxn, i] - g.SR[i])

            WL = g.VL[rho, i]*(g.VL[vxn, i] - g.SL[i])
            WR = g.VR[rho, i]*(g.VR[vxn, i] - g.SR[i])

            VS = (QR - QL)/(WR - WL)

            USL[rho] = g.UL[rho, i]*(g.SL[i] - g.VL[vxn, i])/(g.SL[i] - VS)
            USR[rho] = g.UR[rho, i]*(g.SR[i] - g.VR[vxn, i])/(g.SR[i] - VS)

            USL[mxn] = USL[rho]*VS
            USR[mxn] = USR[rho]*VS
            if p['Dimensions'] == '2D':
                USL[mxt] = USL[rho]*g.VL[vxt, i]
                USR[mxt] = USR[rho]*g.VR[vxt, i]

            USL[eng] = g.UL[eng, i]/g.VL[rho, i] \
                       + (VS - g.VL[vxn, i])*(VS + g.VL[prs, i]/(g.VL[rho, i]*(g.SL[i] - g.VL[vxn, i])))
            USR[eng] = g.UR[eng, i]/g.VR[rho, i] \
                       + (VS - g.VR[vxn, i])*(VS + g.VR[prs, i]/(g.VR[rho, i]*(g.SR[i] - g.VR[vxn, i])))

            USL[eng] *= USL[rho]
            USR[eng] *= USR[rho]

            if VS >= 0.0:
                g.flux[:, i] = g.FL[:, i] + g.SL[i]*(USL - g.UL[:, i])
            else:
                g.flux[:, i] = g.FR[:, i] + g.SR[i]*(USR - g.UR[:, i])

    return
 



