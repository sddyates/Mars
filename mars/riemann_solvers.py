
import numba as nb
import numpy as np
from numba import prange
import sys

from settings import *


#@nb.jit(cache=True, parallel=False)
def tvdlf(FL, FR, UL, UR, VL, VR,
    speed_max, gamma, dtdx,
    vxn, vxt, vxb):
    """
    Synopsis
    --------
    Obtain the fluxes through the upper and
    lower faces of every cell in the column.
    These fluxes are then differenced to give
    the net flux into/out off each cell.

    This function impliments the total variational
    diminishing Lax-Fredrich (tvdlf) Riemann solver.

    Args
    ----
    FL, FR: numpy.array-like
        Array of fluxes on the left and right
        of each cell.

    UL, UR: numpy.array-like
        Array of conserved variables on the left
        and right of each cell face.

    VL, VR: numpy.array-like
        Array of primative variables on the left
        and right of each cell face.

    max_speed: float64-like
        Maximum speed in the domain. Used to calculate
        the time step.

    gamma: numpy.float64-like
        Ratio of specific heats

    dtdx: numpy.float64-like
        Ratio of time and space deltas.

    vxn, vxt, vxb: numpy.int8-like
        index representing the normal, tangential and
        bitangential velocity components relative to the
        sweep direction.

    Attributes
    ----------
    None

    TODO
    ----
    None
    """

    VLR = 0.5*(VL + VR)
    VLR[vxn] = 0.5*(np.absolute(VL[vxn]) + np.absolute(VR[vxn]))

    csLR = np.sqrt(gamma*VLR[prs]/VLR[rho])

    Smax = np.maximum(np.absolute(VLR[vxn] + csLR),
                      np.absolute(VLR[vxn] - csLR))
    flux = 0.5*(FL + FR - Smax*(UR - UL))
    pres = 0.5*(VL[prs] + VR[prs])

    if Smax.max() > speed_max:
        speed_max = Smax.max()

    dflux = -(flux[:, 1:] - flux[:, :-1])*dtdx
    dflux[vxn, :] -= (pres[1:] - pres[:-1])*dtdx

    return dflux, speed_max


#@nb.jit(cache=True, parallel=False)
def hll(FL, FR, UL, UR, VL, VR,
    speed_max, gamma, dtdx,
    vxn, vxt, vxb):

    """
    Synopsis
    --------
    Obtain the fluxes through the upper and
    lower faces of every cell in the column.
    These fluxes are then differenced to give
    the net flux into/out off each cell.

    This function impliments the Harten, Lax
    and van Leer (hll) Riemann solver.

    Args
    ----
    FL, FR: numpy.array-like
        Array of fluxes on the left and right
        of each cell.

    UL, UR: numpy.array-like
        Array of conserved variables on the left
        and right of each cell face.

    VL, VR: numpy.array-like
        Array of primative variables on the left
        and right of each cell face.

    max_speed: float64-like
        Maximum speed in the domain. Used to calculate
        the time step.

    gamma: numpy.float64-like
        Ratio of specific heats

    dtdx: numpy.float64-like
        Ratio of time and space deltas.

    vxn, vxt, vxb: numpy.int8-like
        index representing the normal, tangential and
        bitangential velocity components relative to the
        sweep direction.

    Attributes
    ----------
    None

    TODO
    ----
    None
    """

    flux = np.zeros(shape=FL.shape, dtype=np.float64)
    pres = np.zeros(shape=FL.shape[1], dtype=np.float64)

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

    if np.max(scrh) > speed_max:
        speed_max = np.max(scrh)

    imax = flux.shape[1]

    for i in range(imax):

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

    dflux = -(flux[:, 1:] - flux[:, :-1])*dtdx
    dflux[vxn, :] -= (pres[1:] - pres[:-1])*dtdx

    return dflux, speed_max


#@profile
#@nb.jit(cache=True, parallel=False)
def hllc(FL, FR, UL, UR, VL, VR, speed_max, gamma, dtdx, vxn, vxt, vxb):

    """
    Synopsis
    --------
    Obtain the fluxes through the upper and
    lower faces of every cell in the column.
    These fluxes are then differenced to give
    the net flux into/out off each cell.

    This function impliments the Harten, Lax
    and van Leer Contact (hllc) Riemann solver.

    Args
    ----
    FL, FR: numpy.array-like
        Array of fluxes on the left and right
        of each cell.

    UL, UR: numpy.array-like
        Array of conserved variables on the left
        and right of each cell face.

    VL, VR: numpy.array-like
        Array of primative variables on the left
        and right of each cell face.

    max_speed: float64-like
        Maximum speed in the domain. Used to calculate
        the time step.

    gamma: numpy.float64-like
        Ratio of specific heats

    dtdx: numpy.float64-like
        Ratio of time and space deltas.

    vxn, vxt, vxb: numpy.int8-like
        index representing the normal, tangential and
        bitangential velocity components relative to the
        sweep direction.

    Attributes
    ----------
    None

    TODO
    ----
    None
    """

    usL = np.zeros(FL.shape[0], dtype=np.float64)
    usR = np.zeros(FL.shape[0], dtype=np.float64)
    flux = np.zeros(shape=FL.shape, dtype=np.float64)
    pres = np.zeros(shape=FL.shape[1], dtype=np.float64)

    mxn = vxn
    mxt = vxt
    mxb = vxb

    # Estimate the leftmost and rightmost wave signal
    # speeds bounding the Riemann fan based on the
    # input states VL and VR accourding to the Davis
    # Method.
    temp = gamma*VL[prs, :]/VL[rho, :]
    csL = np.sqrt(temp)
    #
    # print(temp.any() < 0.0)
    # print(np.isnan(temp.any()))

    # if (temp.any() < 0.0) or np.isnan(temp.any()):
    #     print(temp)
    #     sys.exit()

    #print("VL=", VL[prs]/VL[rho])
    #sys.exit()

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            temp = gamma*VL[prs, :]/VL[rho, :]
            csL = np.sqrt(temp)
        except Warning as e:
            print('error found:', e)
            print("VL=", VL[rho], VL[prs])
            sys.exit()

    sL_min = VL[vxn, :] - csL
    sL_max = VL[vxn, :] + csL

    csR = np.sqrt(gamma*VR[prs, :]/VR[rho, :])
    sR_min = VR[vxn, :] - csR
    sR_max = VR[vxn, :] + csR

    SL = np.minimum(sL_min, sR_min)
    SR = np.maximum(sL_max, sR_max)

    scrh = np.maximum(np.absolute(SL),
                      np.absolute(SR))

    if scrh.max() > speed_max:
        speed_max = scrh.max()

    #print(nb.typeof(flux.shape[1]))

    vars = flux.shape[0]
    imax = flux.shape[1]

    for i in range(imax):

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
            if vars > 3:
                usL[mxt] = usL[rho]*vL[vxt]
                usR[mxt] = usR[rho]*vR[vxt]
            if vars > 4:
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

    dflux = -(flux[:, 1:] - flux[:, :-1])*dtdx
    dflux[vxn, :] -= (pres[1:] - pres[:-1])*dtdx

    return dflux, speed_max
