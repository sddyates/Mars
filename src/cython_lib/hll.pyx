import numpy as np
cimport numpy as np

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

