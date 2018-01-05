import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def hll(np.ndarray[DTYPE_t, ndim=2] flux, 
        np.ndarray[DTYPE_t, ndim=1] SL, 
        np.ndarray[DTYPE_t, ndim=1] SR, 
        np.ndarray[DTYPE_t, ndim=2] FL, 
        np.ndarray[DTYPE_t, ndim=2] FR, 
        np.ndarray[DTYPE_t, ndim=2] UL, 
        np.ndarray[DTYPE_t, ndim=2] UR):

    cdef int nvar = flux.shape[0]
    cdef int fmax = flux.shape[1]
    cdef int var, i

    for var in range(nvar):
        for i in range(fmax):
            if SL[i] > 0.0:
                flux[var, i] = FL[var, i]
            elif (SR[i] < 0.0):
                flux[var, i] = FR[var, i]
            else:
                flux[var, i] = (SR[i]*FL[var, i] \
                               - SL[i]*FR[var, i] \
                               + SL[i]*SR[i]*(UR[var, i] \
                               - UL[var, i]))/(SR[i] - SL[i])
    return


@cython.boundscheck(False)
@cython.wraparound(False)
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
    cdef int nvar = flux.shape[0]
    cdef int fmax = flux.shape[1]

    cdef int rho=0, prs=1, vx1=2, vx2=3
    cdef int eng=1, mv1=2, mv2=3 
    cdef int mxn, mxt
    cdef int vxn, vxt

    cdef double QL, QR
    cdef double WL, WR
    cdef double SV

    cdef np.ndarray[DTYPE_t, ndim=2] USL = np.zeros([nvar, fmax], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] USR = np.zeros([nvar, fmax], dtype=DTYPE)

    if p['Dimensions'] == '1D':
        mxn = mv1
        vxn = vx1
    elif p['Dimensions'] == '2D':
        mxn = mv1 if axis == 'i' else mv2
        mxt = mv2 if axis == 'i' else mv1
        vxn = vx1 if axis == 'i' else vx2
        vxt = vx2 if axis == 'i' else vx1

    for i in range(fmax):

        if SL[i] > 0.0:             
            for var in range(nvar):
                flux[var, i] = FL[var, i]

        elif SR[i] < 0.0:
            for var in range(nvar):
                flux[var, i] = FR[var, i]

        else:
            QL = VL[prs, i] + UL[mxn, i]*(VL[vxn, i] - SL[i])
            QR = VR[prs, i] + UR[mxn, i]*(VR[vxn, i] - SR[i])

            WL = VL[rho, i]*(VL[vxn, i] - SL[i])
            WR = VR[rho, i]*(VR[vxn, i] - SR[i])

            VS = (QR - QL)/(WR - WL)

            USL[rho] = UL[rho, i]*(SL[i] - VL[vxn, i])/(SL[i] - VS)
            USR[rho] = UR[rho, i]*(SR[i] - VR[vxn, i])/(SR[i] - VS)

            USL[mxn] = USL[rho]*VS
            USR[mxn] = USR[rho]*VS
            if p['Dimensions'] == '2D':
                USL[mxt] = USL[rho]*VL[vxt, i]
                USR[mxt] = USR[rho]*VR[vxt, i]

            USL[eng] = UL[eng, i]/VL[rho, i] \
                       + (VS - VL[vxn, i])*(VS + VL[prs, i]/(VL[rho, i]*(SL[i] - VL[vxn, i])));
            USR[eng] = UR[eng, i]/VR[rho, i] \
                       + (VS - VR[vxn, i])*(VS + VR[prs, i]/(VR[rho, i]*(SR[i] - VR[vxn, i])));

            USL[eng] *= USL[rho];
            USR[eng] *= USR[rho];

            for var in range(nvar):
                if VS >= 0.0:
                    flux[var, i] = FL[var, i] + SL[i]*(USL[var, i] - UL[var, i]);
                else:
                    flux[var, i] = FR[var, i] + SR[i]*(USR[var, i] - UR[var, i]);

    return
 
