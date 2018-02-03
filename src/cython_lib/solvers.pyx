
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def tvdlf(s):
    Smax = max(np.amax(abs(s.SL)), np.amax(abs(s.SR)))
    return 0.5*(s.FL + s.FR - Smax*(s.UR - s.UL))


def hll(np.ndarray[DTYPE_t, ndim=2] flux, 
        np.ndarray[DTYPE_t, ndim=1] SL, 
        np.ndarray[DTYPE_t, ndim=1] SR, 
        np.ndarray[DTYPE_t, ndim=2] FL, 
        np.ndarray[DTYPE_t, ndim=2] FR, 
        np.ndarray[DTYPE_t, ndim=2] UL, 
        np.ndarray[DTYPE_t, ndim=2] UR):

    cdef int nvar = flux.shape[1]
    cdef int fmax = flux.shape[0]
    cdef int var, i

    for i in range(fmax):
        for var in range(nvar):
            if SL[i] > 0.0:
                flux[i, var] = FL[i, var]
            elif (SR[i] < 0.0):
                flux[i, var] = FR[i, var]
            else:
                flux[i, var] = (SR[i]*FL[i, var] \
                               - SL[i]*FR[i, var] \
                               + SL[i]*SR[i]*(UR[i, var] \
                               - UL[i, var]))/(SR[i] - SL[i])
    return


#@cython.boundscheck(False)
#@cython.wraparound(False)
def hllc(np.ndarray[DTYPE_t, ndim=2] flux, 
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
    cdef int eng=1, mv1=2, mv2=3, mv3=4 
    cdef int mxn, mxt, mxb
    cdef int vxn, vxt, vxb

    cdef double QL, QR
    cdef double WL, WR
    cdef double SV

    cdef np.ndarray[DTYPE_t, ndim=1] USL = np.zeros([nvar], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] USR = np.zeros([nvar], dtype=DTYPE)

    if p['Dimensions'] == '1D':
        mxn = mv1
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

        if SL[i] > 0.0:             
            for var in range(nvar):
                flux[i, var] = FL[i, var]

        elif SR[i] < 0.0:
            for var in range(nvar):
                flux[i, var] = FR[i, var]

        else:
            QL = VL[i, prs] + UL[i, mxn]*(VL[i, vxn] - SL[i])
            QR = VR[i, prs] + UR[i, mxn]*(VR[i, vxn] - SR[i])

            WL = VL[i, rho]*(VL[i, vxn] - SL[i])
            WR = VR[i, rho]*(VR[i, vxn] - SR[i])

            VS = (QR - QL)/(WR - WL)

            USL[rho] = UL[i, rho]*(SL[i] - VL[i, vxn])/(SL[i] - VS)
            USR[rho] = UR[i, rho]*(SR[i] - VR[i, vxn])/(SR[i] - VS)

            USL[mxn] = USL[rho]*VS
            USR[mxn] = USR[rho]*VS
            if p['Dimensions'] == '2D':
                USL[mxt] = USL[rho]*VL[i, vxt]
                USR[mxt] = USR[rho]*VR[i, vxt]
            if p['Dimensions'] == '3D':
                USL[mxb] = USL[rho]*VL[i, vxb]
                USR[mxb] = USR[rho]*VR[i, vxb]

            USL[eng] = UL[i, eng]/VL[i, rho] \
                        + (VS - VL[i, vxn])*(VS \
                            + VL[i, prs]/(VL[i, rho]*(SL[i] - VL[i, vxn])))
            USR[eng] = UR[i, eng]/VR[i, rho] \
                        + (VS - VR[i, vxn])*(VS \
                            + VR[i, prs]/(VR[i, rho]*(SR[i] - VR[i, vxn])))

            USL[eng] *= USL[rho];
            USR[eng] *= USR[rho];

            for var in range(nvar):
                if VS >= 0.0:
                    flux[i, var] = FL[i, var] + SL[i]*(USL[var] - UL[i, var]);
                else:
                    flux[i, var] = FR[i, var] + SR[i]*(USR[var] - UR[i, var]);

    return
 
