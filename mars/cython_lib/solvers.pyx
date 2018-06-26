
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def tvdlf(g):
    rho = 0
    prs = 1
    vx1 = 2
    VLR = 0.5*(g.VL + g.VR)
    csLR = np.sqrt(1.666666*VLR[prs]/VLR[rho])
    Smax = np.maximum(abs(VLR[vx1] + csLR), abs(VLR[vx1] - csLR))

    #Smax = max(np.amax(abs(g.SL)), np.amax(abs(g.SR)))
    g.flux = 0.5*(g.FL + g.FR - Smax*(g.UR - g.UL))
    g.pres = 0.5*(g.VL[prs] + g.VL[prs])
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
        np.ndarray[DTYPE_t, ndim=2] VR):

    cdef int nvar = flux.shape[1]
    cdef int fmax = flux.shape[0]
    cdef int var, i, prs = 1
    cdef double scrh

    for i in range(fmax):
        for var in range(nvar):
            if SL[i] > 0.0:
                flux[i, var] = FL[i, var]
                pres[i] = VL[i, prs]
            elif (SR[i] < 0.0):
                flux[i, var] = FR[i, var]
                pres[i] = VR[i, prs]
            else:
                scrh = 1.0 / (SR[i] - SL[i])
                flux[i, var] = SL[i]*SR[i]*(UR[i, var] - UL[i, var]) \
                    + SR[i]*FL[i, var] - SL[i]*FR[i, var]
                flux[i, var] *= scrh
                pres[i] = (SR[i]*VL[i, prs] - SL[i]*VR[i, prs])*scrh
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

    cdef double QL, QR
    cdef double WL, WR
    cdef double SV

    cdef np.ndarray[DTYPE_t, ndim=1] USL = np.zeros([nvar], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] USR = np.zeros([nvar], dtype=DTYPE)

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

            #VS = VR[i, prs] - VL[i, prs] + UL[i, mxn]*(SL[i] - VL[i, vxn]) \
            #    - UR[i, mxn]*(SR[i] - VR[i, vxn])
            #VS /= VL[i, rho]*(SL[i] - VL[i, vxn]) - VR[i, rho]*(SR[i] - VR[i, vxn])

            USL[rho] = UL[i, rho]*(SL[i] - VL[i, vxn])/(SL[i] - VS)
            USR[rho] = UR[i, rho]*(SR[i] - VR[i, vxn])/(SR[i] - VS)

            if p['Dimensions'] == '1D':
                USL[mxn] = USL[rho]*VS
                USR[mxn] = USR[rho]*VS
            elif p['Dimensions'] == '2D':
                USL[mxn] = USL[rho]*VS
                USR[mxn] = USR[rho]*VS
                USL[mxt] = USL[rho]*VL[i, vxt]
                USR[mxt] = USR[rho]*VR[i, vxt]
            elif p['Dimensions'] == '3D':
                USL[mxn] = USL[rho]*VS
                USR[mxn] = USR[rho]*VS
                USL[mxt] = USL[rho]*VL[i, vxt]
                USR[mxt] = USR[rho]*VR[i, vxt]
                USL[mxb] = USL[rho]*VL[i, vxb]
                USR[mxb] = USR[rho]*VR[i, vxb]

            USL[eng] = UL[i, eng]/VL[i, rho] \
                        + (VS - VL[i, vxn])*(VS \
                        + VL[i, prs]/(VL[i, rho]*(SL[i] - VL[i, vxn])))
            USR[eng] = UR[i, eng]/VR[i, rho] \
                        + (VS - VR[i, vxn])*(VS \
                        + VR[i, prs]/(VR[i, rho]*(SR[i] - VR[i, vxn])))

            USL[eng] *= USL[rho]
            USR[eng] *= USR[rho]

            for var in range(nvar):
                if VS >= 0.0:
                    flux[i, var] = FL[i, var] + SL[i]*(USL[var] - UL[i, var])
                    pres[i] = VL[i, prs]
                else:
                    flux[i, var] = FR[i, var] + SR[i]*(USR[var] - UR[i, var])
                    pres[i] = VR[i, prs]

    return
 
