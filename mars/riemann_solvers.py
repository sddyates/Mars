
from numba import jit
import numpy as np
from settings import *


@jit
def tvdlf(g, a, vxn, vxt, vxb):

    VLR = 0.5*(g.VL + g.VR)
    VLR[vxn] = 0.5*(abs(g.VL[vxn]) + abs(g.VR[vxn]))

    csLR = np.sqrt(a.gamma*VLR[prs]/VLR[rho])

    Smax = np.maximum(np.absolute(VLR[vxn] + csLR),
                      np.absolute(VLR[vxn] - csLR))
    g.flux = 0.5*(g.FL + g.FR - Smax*(g.UR - g.UL))
    g.pres = 0.5*(g.VL[prs] + g.VR[prs])

    return


@jit
def hll(g, a, vxn, vxt, vxb):

    # Estimate the leftmost and rightmost wave signal
    # speeds bounding the Riemann fan based on the
    # input states VL and VR accourding to the Davis
    # Method.
    csL = np.sqrt(a.gamma*g.VL[prs, :]/g.VL[rho, :])
    sL_min = g.VL[vxn, :] - csL
    sL_max = g.VL[vxn, :] + csL

    csR = np.sqrt(a.gamma*g.VR[prs, :]/g.VR[rho, :])
    sR_min = g.VR[vxn, :] - csR
    sR_max = g.VR[vxn, :] + csR

    g.SL = np.minimum(sL_min, sR_min)
    g.SR = np.maximum(sL_max, sR_max)

    scrh = np.maximum(np.absolute(g.SL),
                      np.absolute(g.SR))
    g.cmax = scrh

    for i in range(g.flux.shape[1]):

        if g.SL[i] > 0.0:
            g.flux[:, i] = g.FL[:, i]
            g.pres[i] = g.VL[prs, i]

        elif (g.SR[i] < 0.0):
            g.flux[:, i] = g.FR[:, i]
            g.pres[i] = g.VR[prs, i]

        else:
            scrh[i] = 1.0/(g.SR[i] - g.SL[i])
            g.flux[:, i] = g.SL[i]*g.SR[i]*(g.UR[:, i] - g.UL[:, i]) \
                + g.SR[i]*g.FL[:, i] - g.SL[i]*g.FR[:, i]
            g.flux[:, i] *= scrh[i]
            g.pres[i] = (g.SR[i]*g.VL[prs, i] - g.SL[i]*g.VR[prs, i])*scrh[i]

    return


# @profile
@jit
def hllc(g, a, vxn, vxt, vxb):

    mxn = vxn
    mxt = vxt
    mxb = vxb

    usL = np.zeros([g.flux.shape[0]], dtype=np.float64)
    usR = np.zeros([g.flux.shape[0]], dtype=np.float64)

    # Estimate the leftmost and rightmost wave signal
    # speeds bounding the Riemann fan based on the
    # input states VL and VR accourding to the Davis
    # Method.
    csL = np.sqrt(a.gamma*g.VL[prs, :]/g.VL[rho, :])
    sL_min = g.VL[vxn, :] - csL
    sL_max = g.VL[vxn, :] + csL

    csR = np.sqrt(a.gamma*g.VR[prs, :]/g.VR[rho, :])
    sR_min = g.VR[vxn, :] - csR
    sR_max = g.VR[vxn, :] + csR

    g.SL = np.minimum(sL_min, sR_min)
    g.SR = np.maximum(sL_max, sR_max)

    scrh = np.maximum(np.absolute(g.SL),
                      np.absolute(g.SR))
    g.cmax  = scrh

    for i in range(g.flux.shape[1]):

        if g.SL[i] > 0.0:
            g.flux[:, i] = g.FL[:, i]
            g.pres[i] = g.VL[prs, i]

        elif (g.SR[i] < 0.0):
            g.flux[:, i] = g.FR[:, i]
            g.pres[i] = g.VR[prs, i]

        else:

            vR = g.VR[:, i]
            uR = g.UR[:, i]

            vL = g.VL[:, i]
            uL = g.UL[:, i]

            vxr = vR[vxn]
            vxl = vL[vxn]

            qL = vL[prs] + uL[mxn]*(vL[vxn] - g.SL[i])
            qR = vR[prs] + uR[mxn]*(vR[vxn] - g.SR[i])

            wL = vL[rho]*(vL[vxn] - g.SL[i])
            wR = vR[rho]*(vR[vxn] - g.SR[i])

            vs = (qR - qL)/(wR - wL)

            usL[rho] = uL[rho]*(g.SL[i] - vxl)/(g.SL[i] - vs)
            usR[rho] = uR[rho]*(g.SR[i] - vxr)/(g.SR[i] - vs)

            usL[mxn] = usL[rho]*vs
            usR[mxn] = usR[rho]*vs
            if a.is_2D:
                usL[mxt] = usL[rho]*vL[vxt]
                usR[mxt] = usR[rho]*vR[vxt]
            if a.is_3D:
                usL[mxt] = usL[rho]*vL[vxt]
                usR[mxt] = usR[rho]*vR[vxt]
                usL[mxb] = usL[rho]*vL[vxb]
                usR[mxb] = usR[rho]*vR[vxb]

            usL[eng] = uL[eng]/vL[rho] \
                       + (vs - vxl)*(vs + vL[prs]/(vL[rho]*(g.SL[i] - vxl)))
            usR[eng] = uR[eng]/vR[rho] \
                       + (vs - vxr)*(vs + vR[prs]/(vR[rho]*(g.SR[i] - vxr)))

            usL[eng] *= usL[rho]
            usR[eng] *= usR[rho]

            if (vs >= 0.0):
                g.flux[:, i] = g.FL[:, i] + g.SL[i]*(usL[:] - uL[:])
                g.pres[i] = g.VL[prs, i]

            else:
                g.flux[:, i] = g.FR[:, i] + g.SR[i]*(usR[:] - uR[:])
                g.pres[i] = g.VR[prs, i]

    return
