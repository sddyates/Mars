
import numpy as np
cimport numpy as np
from settings import *
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def tvdlf_pyx(np.ndarray[DTYPE_t, ndim=2] FL,
          np.ndarray[DTYPE_t, ndim=2] FR,
          np.ndarray[DTYPE_t, ndim=2] UL,
          np.ndarray[DTYPE_t, ndim=2] UR,
          np.ndarray[DTYPE_t, ndim=2] VL,
          np.ndarray[DTYPE_t, ndim=2] VR,
          cdef DTYPE_t speed_max,
          cdef DTYPE_t gamma,
          cdef DTYPE_t dtdx,
          cdef int vxn,
          cdef int vxt,
          cdef int vxb):
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



def hll_pyx(np.ndarray[DTYPE_t, ndim=2] FL,
        np.ndarray[DTYPE_t, ndim=2] FR,
        np.ndarray[DTYPE_t, ndim=2] UL,
        np.ndarray[DTYPE_t, ndim=2] UR,
        np.ndarray[DTYPE_t, ndim=2] VL,
        np.ndarray[DTYPE_t, ndim=2] VR,
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

    cdef int imax = FL.shape[1]
    cdef int nvar = FL.shape[0]
    cdef int var, i
    cdef int rho = 0, prs = 1
    cdef int vx1 = 2, vx2 = 3, vx3 = 4
    cdef int mvx1 = 2, mvx2 = 3, mvx3 = 4

    cdef np.ndarray[DTYPE_t, ndim=1] scrh = np.empty([imax], dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1] flux = np.empty([nvar, imax], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] pres = np.empty([imax], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] dflux = np.empty([nvar, imax-1], dtype=DTYPE)


    # Estimate the leftmost and rightmost wave signal
    # speeds bounding the Riemann fan based on the
    # input states VL and VR accourding to the Davis
    # Method.
    for i in range(imax):
        csL = np.sqrt(gamma*VL[prs, i]/VL[rho, i])
        sL_min = VL[vxn, i] - csL
        sL_max = VL[vxn, i] + csL

        csR = np.sqrt(gamma*VR[prs, i]/VR[rho, i])
        sR_min = VR[vxn, i] - csR
        sR_max = VR[vxn, i] + csR

        SL[i] = np.minimum(sL_min, sR_min)
        SR[i] = np.maximum(sL_max, sR_max)

        scrh[i] = np.maximum(np.absolute(SL[i]),
                             np.absolute(SR[i]))

        if scrh[i] > speed_max:
            speed_max = np.max(scrh)

    imax = flux.shape[1]

    for i in range(imax):

        if SL[i] > 0.0:
            for var in range(nvar):
                flux[var, i] = FL[var, i]
            pres[i] = VL[prs, i]

        elif (SR[i] < 0.0):
            for var in range(nvar):
                flux[var, i] = FR[var, i]
            pres[i] = VR[prs, i]

        else:
            scrh[i] = 1.0/(SR[i] - SL[i])
            for var in range(nvar):
                flux[var, i] = SL[i]*SR[i]*(UR[var, i] - UL[var, i]) \
                    + SR[i]*FL[var, i] - SL[i]*FR[var, i]
                flux[var, i] *= scrh[i]
            pres[i] = (SR[i]*VL[prs, i] - SL[i]*VR[prs, i])*scrh[i]

    for i in range(imax-1):
        for var in range(nvar):
            dflux = -(flux[var, i+1] - flux[var, i])*dtdx
        dflux[vxn, i] -= (pres[i+1] - pres[i])*dtdx

    return dflux, speed_max


def hllc_pyx(FL, FR, UL, UR, VL, VR,
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

    cdef int i, var
    cdef int nvar = FL.shape[1]
    cdef int imax = FL.shape[0]

    cdef int rho=0, prs=1, vx1=2, vx2=3, vx3=4
    cdef int eng=1, mvx1=2, mvx2=3, mvx3=4
    cdef int mxn, mxt, mxb

    cdef DTYPE_t csL, csR
    cdef DTYPE_t sL_min, sL_max
    cdef DTYPE_t sR_min, sR_max

    cdef DTYPE_t vxr
    cdef DTYPE_t vxl

    cdef DTYPE_t qL
    cdef DTYPE_t qR

    cdef DTYPE_t wL
    cdef DTYPE_t wR

    cdef np.ndarray[DTYPE_t, ndim=1] vR = np.empty([imax], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] uR = np.empty([imax], dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1] vL = np.empty([imax], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] uL = np.empty([imax], dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1] usL = np.empty([nvar], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] usR = np.empty([nvar], dtype=DTYPE)

    usL = np.empty(FL.shape[0], dtype=np.float64)
    usR = np.empty(FL.shape[0], dtype=np.float64)
    flux = np.empty(shape=FL.shape, dtype=np.float64)
    pres = np.empty(shape=FL.shape[1], dtype=np.float64)

    mxn = vxn
    mxt = vxt
    mxb = vxb

    # Estimate the leftmost and rightmost wave signal
    # speeds bounding the Riemann fan based on the
    # input states VL and VR accourding to the Davis
    # Method.
    for i in range(imax):
        csL = np.sqrt(gamma*VL[prs, i]/VL[rho, i])
        sL_min = VL[vxn, i] - csL
        sL_max = VL[vxn, i] + csL

        csR = np.sqrt(gamma*VR[prs, i]/VR[rho, i])
        sR_min = VR[vxn, i] - csR
        sR_max = VR[vxn, i] + csR

        SL[i] = np.minimum(sL_min, sR_min)
        SR[i] = np.maximum(sL_max, sR_max)

        scrh[i] = np.maximum(np.absolute(SL[i]),
                             np.absolute(SR[i]))

        if scrh[i] > speed_max:
            speed_max = np.max(scrh)


    vars = flux.shape[0]
    imax = flux.shape[1]

    for i in range(imax):

        if SL[i] > 0.0:
            for var in range(nvar):
                flux[var, i] = FL[var, i]
            pres[i] = VL[prs, i]

        elif (SR[i] < 0.0):
            for var in range(nvar):
                flux[var, i] = FR[var, i]
            pres[i] = VR[prs, i]

        else:

            for var in range(nvar):
                vR[var] = VR[var, i]
                uR[var] = UR[var, i]

                vL[var] = VL[var, i]
                uL[var] = UL[var, i]

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
                for var in range(nvar):
                    flux[var, i] = FL[var, i] + SL[i]*(usL[var] - uL[var])
                pres[i] = VL[prs, i]

            else:
                flux[var, i] = FR[var, i] + SR[i]*(usR[var] - uR[var])
                pres[i] = VR[prs, i]

    for i in range(imax-1):
        for var in range(nvar):
            dflux = -(flux[var, i+1] - flux[var, i])*dtdx
        dflux[vxn, i] -= (pres[i+1] - pres[i])*dtdx

    return dflux, speed_max
