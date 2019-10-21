
from numba import jit
import numba as nb
import numpy as np
from settings import *


@jit(cache=True)
def tvdlf(flux, FL, FR, UL, UR, VL, VR, SL, SR, pres, cs_max, speed_max, gamma, vxn, vxt, vxb):

    VLR = 0.5*(VL + VR)
    VLR[vxn] = 0.5*(np.absolute(VL[vxn]) + np.absolute(VR[vxn]))

    csLR = np.sqrt(gamma*VLR[prs]/VLR[rho])

    Smax = np.maximum(np.absolute(VLR[vxn] + csLR),
                      np.absolute(VLR[vxn] - csLR))
    flux = 0.5*(FL + FR - Smax*(UR - UL))
    pres = 0.5*(VL[prs] + VR[prs])

    cs_max = np.absolute(csLR).max()
    if cs_max > cs_max:
        cs_max = cs_max

    if Smax.max() > speed_max:
        speed_max = Smax.max()

    return flux, pres, cs_max, speed_max


@nb.jit(cache=True)
def hll(flux, FL, FR, UL, UR, VL, VR, SL, SR, pres, cs_max, speed_max, gamma, vxn, vxt, vxb):

    # Estimate the leftmost and rightmost wave signal
    # speeds bounding the Riemann fan based on the
    # input states VL and VR accourding to the Davis
    # Method.
    csL = np.sqrt(gamma*VL[prs, :]/VL[rho, :])
    sL_min = VL[vxn, :] - csL
    sL_max = VL[vxn, :] + csL

    csR = np.sqrt(gamma*VR[prs, :]/VR[rho, :])
    sR_min = VR[vxn, :] - csR
    sR_max = VR[vxn, :] + csR

    SL = np.minimum(sL_min, sR_min)
    SR = np.maximum(sL_max, sR_max)

    scrh = np.maximum(np.absolute(SL),
                      np.absolute(SR))

    cs_max = np.maximum(np.absolute(csL), np.absolute(csR)).max()
    if cs_max > cs_max:
        cs_max = cs_max

    if np.max(scrh) > speed_max:
        speed_max = np.max(scrh)

    for i in range(flux.shape[1]):

        if SL[i] > 0.0:
            flux[:, i] = FL[:, i]
            pres[i] = VL[prs, i]

        elif (SR[i] < 0.0):
            flux[:, i] = FR[:, i]
            pres[i] = VR[prs, i]

        else:
            scrh[i] = 1.0/(SR[i] - SL[i])
            flux[:, i] = SL[i]*SR[i]*(UR[:, i] - UL[:, i]) \
                + SR[i]*FL[:, i] - SL[i]*FR[:, i]
            flux[:, i] *= scrh[i]
            pres[i] = (SR[i]*VL[prs, i] - SL[i]*VR[prs, i])*scrh[i]

    return flux, pres, cs_max, speed_max


#@profile
@jit(cache=True)
def hllc(flux, FL, FR, UL, UR, VL, VR, SL, SR, pres, cs_max, speed_max, gamma, vxn, vxt, vxb):

    mxn = vxn
    mxt = vxt
    mxb = vxb

    usL = np.zeros(flux.shape[0], dtype=np.float64)
    usR = np.zeros(flux.shape[0], dtype=np.float64)

    # Estimate the leftmost and rightmost wave signal
    # speeds bounding the Riemann fan based on the
    # input states VL and VR accourding to the Davis
    # Method.
    csL = np.sqrt(gamma*VL[prs, :]/VL[rho, :])
    sL_min = VL[vxn, :] - csL
    sL_max = VL[vxn, :] + csL

    csR = np.sqrt(gamma*VR[prs, :]/VR[rho, :])
    sR_min = VR[vxn, :] - csR
    sR_max = VR[vxn, :] + csR

    SL = np.minimum(sL_min, sR_min)
    SR = np.maximum(sL_max, sR_max)

    scrh = np.maximum(np.absolute(SL),
                      np.absolute(SR))

    cs_max = np.maximum(np.absolute(csL), np.absolute(csR)).max()
    if cs_max > cs_max:
        cs_max = cs_max

    if scrh.max() > speed_max:
        speed_max = scrh.max()

    for i in range(flux.shape[1]):

        if SL[i] > 0.0:
            flux[:, i] = FL[:, i]
            pres[i] = VL[prs, i]

        elif (SR[i] < 0.0):
            flux[:, i] = FR[:, i]
            pres[i] = VR[prs, i]

        else:

            vR = VR[:, i]
            uR = UR[:, i]

            vL = VL[:, i]
            uL = UL[:, i]

            vxr = vR[vxn]
            vxl = vL[vxn]

            qL = vL[prs] + uL[mxn]*(vL[vxn] - SL[i])
            qR = vR[prs] + uR[mxn]*(vR[vxn] - SR[i])

            wL = vL[rho]*(vL[vxn] - SL[i])
            wR = vR[rho]*(vR[vxn] - SR[i])

            vs = (qR - qL)/(wR - wL)

            usL[rho] = uL[rho]*(SL[i] - vxl)/(SL[i] - vs)
            usR[rho] = uR[rho]*(SR[i] - vxr)/(SR[i] - vs)

            usL[mxn] = usL[rho]*vs
            usR[mxn] = usR[rho]*vs
            if flux.shape[0] > 3:
                usL[mxt] = usL[rho]*vL[vxt]
                usR[mxt] = usR[rho]*vR[vxt]
            if flux.shape[0] > 4:
                usL[mxt] = usL[rho]*vL[vxt]
                usR[mxt] = usR[rho]*vR[vxt]
                usL[mxb] = usL[rho]*vL[vxb]
                usR[mxb] = usR[rho]*vR[vxb]

            usL[eng] = uL[eng]/vL[rho] \
                       + (vs - vxl)*(vs + vL[prs]/(vL[rho]*(SL[i] - vxl)))
            usR[eng] = uR[eng]/vR[rho] \
                       + (vs - vxr)*(vs + vR[prs]/(vR[rho]*(SR[i] - vxr)))

            usL[eng] *= usL[rho]
            usR[eng] *= usR[rho]

            if (vs >= 0.0):
                flux[:, i] = FL[:, i] + SL[i]*(usL[:] - uL[:])
                pres[i] = VL[prs, i]

            else:
                flux[:, i] = FR[:, i] + SR[i]*(usR[:] - uR[:])
                pres[i] = VR[prs, i]

    return flux, pres, cs_max, speed_max
