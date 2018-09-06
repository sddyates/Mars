
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def tvdlf(g, p, axis):
    rho = 0
    prs = 1
    vx1 = 2
    vx2 = 3
    vx3 = 4
    mvx1 = 2
    mvx2 = 3
    mvx3 = 4

    if p['Dimensions'] == '1D':
        mxn = mvx1
        vxn = vx1
    elif p['Dimensions'] == '2D' and axis == 'i':
        mxn = vxn = vx1
        mxt = vxt = vx2
    elif p['Dimensions'] == '2D' and axis == 'j':
        mxn = vxn = vx2
        mxt = vxt = vx1
    elif p['Dimensions'] == '3D' and axis == 'i':
        mxn = vxn = vx1
        mxt = vxt = vx2
        mxb = vxb = vx3
    elif p['Dimensions'] == '3D' and axis == 'j': 
        mxn = vxn = vx2
        mxt = vxt = vx1
        mxb = vxb = vx3
    elif p['Dimensions'] == '3D' and axis == 'k':
        mxn = vxn = vx3
        mxt = vxt = vx1
        mxb = vxb = vx2

    VLR = 0.5*(g.VL + g.VR)
    VLR[vxn] = 0.5*(abs(g.VL[vxn]) + abs(g.VR[vxn]))

    csLR = np.sqrt(p['gamma']*VLR[prs]/VLR[rho])

    Smax = np.maximum(np.absolute(VLR[vxn] + csLR), 
                      np.absolute(VLR[vxn] - csLR))
    g.flux = 0.5*(g.FL + g.FR - Smax*(g.UR - g.UL))
    g.pres = 0.5*(g.VL[prs] + g.VR[prs])

    return


def hll(np.ndarray[DTYPE_t, ndim=2] flux,
        np.ndarray[DTYPE_t, ndim=1] pres,
        np.ndarray[DTYPE_t, ndim=1] SL, 
        np.ndarray[DTYPE_t, ndim=1] SR, 
        np.ndarray[DTYPE_t, ndim=2] FL, 
        np.ndarray[DTYPE_t, ndim=2] FR, 
        np.ndarray[DTYPE_t, ndim=2] UL, 
        np.ndarray[DTYPE_t, ndim=2] UR,
        np.ndarray[DTYPE_t, ndim=2] VL, 
        np.ndarray[DTYPE_t, ndim=2] VR,
        p, 
        axis):

    cdef int imax = flux.shape[0]
    cdef int nvar = flux.shape[1]
    cdef int var, i
    cdef int rho = 0, prs = 1
    cdef int vx1 = 2, vx2 = 3, vx3 = 4
    cdef int mvx1 = 2, mvx2 = 3, mvx3 = 4

    cdef np.ndarray[DTYPE_t, ndim=1] scrh = np.zeros([imax], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] cmax = np.zeros([imax], dtype=DTYPE)

    cdef double sL_min, sL_max, csL
    cdef double sR_min, sR_max, csR
    
    if p['Dimensions'] == '1D':
        mxn = mvx1
        vxn = vx1
    elif p['Dimensions'] == '2D' and axis == 'i':
        mxn = vxn = vx1
        mxt = vxt = vx2
    elif p['Dimensions'] == '2D' and axis == 'j':
        mxn = vxn = vx2
        mxt = vxt = vx1
    elif p['Dimensions'] == '3D' and axis == 'i':
        mxn = vxn = vx1
        mxt = vxt = vx2
        mxb = vxb = vx3
    elif p['Dimensions'] == '3D' and axis == 'j': 
        mxn = vxn = vx2
        mxt = vxt = vx1
        mxb = vxb = vx3
    elif p['Dimensions'] == '3D' and axis == 'k':
        mxn = vxn = vx3
        mxt = vxt = vx1
        mxb = vxb = vx2

    # Estimate the leftmost and rightmost wave signal 
    # speeds bounding the Riemann fan based on the 
    # input states VL and VR accourding to the Davis 
    # Method.
    for i in range(imax):
        csL = np.sqrt(p['gamma']*VL[i, prs]/VL[i, rho])
        sL_min = VL[i, vxn] - csL
        sL_max = VL[i, vxn] + csL

        csR = np.sqrt(p['gamma']*VR[i, prs]/VR[i, rho])
        sR_min = VR[i, vxn] - csR
        sR_max = VR[i, vxn] + csR

        SL[i] = np.minimum(sL_min, sR_min)
        SR[i] = np.maximum(sL_max, sR_max)

        scrh[i] = np.maximum(np.absolute(SL[i]), 
                             np.absolute(SR[i]))
        cmax[i] = scrh[i]

    for i in range(imax):
        if SL[i] > 0.0:

            for var in range(nvar):
                flux[i, var] = FL[i, var]
            pres[i] = VL[i, prs]

        elif (SR[i] < 0.0):

            for var in range(nvar):
                flux[i, var] = FR[i, var]
            pres[i] = VR[i, prs]

        else:

            scrh[i] = 1.0 / (SR[i] - SL[i])
            for var in range(nvar):
                flux[i, var] = SL[i]*SR[i]*(UR[i, var] - UL[i, var]) \
                    + SR[i]*FL[i, var] - SL[i]*FR[i, var]
                flux[i, var] *= scrh[i]
            pres[i] = (SR[i]*VL[i, prs] - SL[i]*VR[i, prs])*scrh[i]

    return


@cython.boundscheck(False)
@cython.wraparound(False)
def hllc(np.ndarray[DTYPE_t, ndim=2] flux, 
         np.ndarray[DTYPE_t, ndim=1] pres,
         np.ndarray[DTYPE_t, ndim=1] SL, 
         np.ndarray[DTYPE_t, ndim=1] SR, 
         np.ndarray[DTYPE_t, ndim=2] FL, 
         np.ndarray[DTYPE_t, ndim=2] FR, 
         np.ndarray[DTYPE_t, ndim=2] UL, 
         np.ndarray[DTYPE_t, ndim=2] UR, 
         np.ndarray[DTYPE_t, ndim=2] VL, 
         np.ndarray[DTYPE_t, ndim=2] VR, 
         p, 
         axis):

    cdef int i, var
    cdef int nvar = flux.shape[1]
    cdef int fmax = flux.shape[0]

    cdef int rho=0, prs=1, vx1=2, vx2=3, vx3=4
    cdef int eng=1, mvx1=2, mvx2=3, mvx3=4 
    cdef int mxn, mxt, mxb
    cdef int vxn, vxt, vxb

    cdef np.ndarray[DTYPE_t, ndim=1] vR = np.zeros([fmax], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] uR = np.zeros([fmax], dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1] vL = np.zeros([fmax], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] uL = np.zeros([fmax], dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1] usL = np.zeros([nvar], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] usR = np.zeros([nvar], dtype=DTYPE)

    cdef double csL, csR
    cdef double sL_min, sL_max
    cdef double sR_min, sR_max

    cdef double vxr
    cdef double vxl

    cdef double qL
    cdef double qR

    cdef double wL
    cdef double wR

    cdef double vs

    if p['Dimensions'] == '1D':
        mxn = mvx1
        vxn = vx1
    elif p['Dimensions'] == '2D' and axis == 'i':
        mxn = vxn = vx1
        mxt = vxt = vx2
    elif p['Dimensions'] == '2D' and axis == 'j':
        mxn = vxn = vx2
        mxt = vxt = vx1
    elif p['Dimensions'] == '3D' and axis == 'i':
        mxn = vxn = vx1
        mxt = vxt = vx2
        mxb = vxb = vx3
    elif p['Dimensions'] == '3D' and axis == 'j': 
        mxn = vxn = vx2
        mxt = vxt = vx1
        mxb = vxb = vx3
    elif p['Dimensions'] == '3D' and axis == 'k':
        mxn = vxn = vx3
        mxt = vxt = vx1
        mxb = vxb = vx2

    for i in range(fmax):

        # Estimate the leftmost and rightmost wave signal 
        # speeds bounding the Riemann fan based on the 
        # input states VL and VR accourding to the Davis 
        # Method.

        csL = np.sqrt(p['gamma']*VL[i, prs]/VL[i, rho])
        sL_min = VL[i, vxn] - csL
        sL_max = VL[i, vxn] + csL

        csR = np.sqrt(p['gamma']*VR[i, prs]/VR[i, rho])
        sR_min = VR[i, vxn] - csR
        sR_max = VR[i, vxn] + csR

        SL[i] = np.minimum(sL_min, sR_min)
        SR[i] = np.maximum(sL_max, sR_max)

    for i in range(fmax):

        scrh = np.maximum(np.absolute(SL[i]), 
                          np.absolute(SR[i]))
        cmax  = scrh

        if (SL[i] > 0.0):
        
            for var in range(nvar):
                flux[i, var] = FL[i, var]
            pres[i] = VL[i, prs]

        elif (SR[i] < 0.0):

            for var in range(nvar):
                flux[i, var] = FR[i, var]
            pres[i] = VR[i, prs]

        else:

            for var in range(nvar):
                vR[var] = VR[i, var]
                uR[var] = UR[i, var]

                vL[var] = VL[i, var]
                uL[var] = UL[i, var]

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
            if p['Dimensions'] == '2D' or '3D':
                usL[mxt] = usL[rho]*vL[vxt]
                usR[mxt] = usR[rho]*vR[vxt]
            if p['Dimensions'] == '3D':
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
                    flux[i, var] = FL[i, var] + SL[i]*(usL[var] - uL[var])
                pres[i] = VL[i, prs]

            else:

                for var in range(nvar):
                    flux[i, var] = FR[i, var] + SR[i]*(usR[var] - uR[var])
                pres[i] = VR[i, prs]

    return