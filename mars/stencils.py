import numpy as np
import numba as nb

from settings import *

@nb.stencil
def OneD_first_order_stencil(U, gamma, gamma_1, igamma_1, speed_max, dtdx, vxn, vxt, vxb):

    V = np.empty(shape=U.shape, dtype=np.float64)

    # Conservitive to primative
    V[rho, 0] = U[rho, 0]

    #irho = np.empty(shape=U.shape[1], dtype=np.float64)
    #m2 = np.empty(shape=U.shape[1], dtype=np.float64)

    irho = 1.0/V[rho, 0]

    m2 = U[mvx1, 0]**2
    V[vx1, 0] = U[mvx1, 0]*irho
    if U.shape[0] > 3:
        m2 += U[mvx2, 0]**2
        V[vx2, 0] = U[mvx2, 0]*irho
    if U.shape[0] > 4:
        m2 += U[mvx3, 0]**2
        V[vx3, 0] = U[mvx3, 0]*irho
    V[prs, 0] = gamma_1*(U[eng, 0] - 0.5*m2*irho)


    RHO = U[rho, 0]

    #irho = np.empty(shape=U.shape[1], dtype=np.float64)
    #m2 = np.empty(shape=U.shape[1], dtype=np.float64)

    iRHO = 1.0/RHO

    m2 = U[mvx1, 0]**2
    VX1 = U[mvx1, 0]*irho
    if U.shape[0] > 3:
        m2 += U[mvx2, 0]**2
        VX2 = U[mvx2, 0]*irho
    if U.shape[0] > 4:
        m2 += U[mvx3, 0]**2
        VX3 = U[mvx3, 0]*irho
    PRS = gamma_1*(U[eng, 0] - 0.5*m2*irho)


    # Reconstruction
    VL, VR = V[0, -1], V[0, 1]

    UL = np.empty((U.shape[0], U.shape[1]-1), dtype=np.float64)
    UR = np.empty((U.shape[0], U.shape[1]-1), dtype=np.float64)

    # Primative to conservative
    #v2 = np.empty(shape=UL.shape[1], dtype=np.float64)
    UL[rho, 0] = VL[rho, 0]
    v2 = VL[vx1, 0]**2
    UL[mvx1, 0] = VL[vx1, 0]*VL[rho, 0]
    if UL.shape[0] > 3:
        v2 += VL[vx2, 0]**2
        UL[mvx2, 0] = VL[vx2, 0]*VL[rho, 0]
    if UL.shape[0] > 4:
        v2 += VL[vx3, 0]**2
        UL[mvx3, 0] = VL[vx3, 0]*VL[rho, 0]
    UL[eng, 0] = 0.5*VL[rho, 0]*v2 + VL[prs, 0]*igamma_1

    UR[rho, 0] = VR[rho, 0]
    v2 = VR[vx1, 0]**2
    UR[mvx1, :] = VR[vx1, 0]*VR[rho, 0]
    if UR.shape[0] > 3:
        v2 += VR[vx2, 0]**2
        UR[mvx2, 0] = VR[vx2, 0]*VR[rho, 0]
    if UR.shape[0] > 4:
        v2 += VR[vx3, 0]**2
        UR[mvx3, 0] = VR[vx3, 0]*VR[rho, 0]
    UR[eng, 0] = 0.5*VR[rho, 0]*v2 + VR[prs, 0]*igamma_1

    FL = np.empty(shape=VL.shape, dtype=np.float64)
    FR = np.empty(shape=VR.shape, dtype=np.float64)

    # Flux tensor
    FL[rho, 0] = UL[vxn, 0]
    FL[eng, 0] = VL[vxn, 0]*(UL[eng, 0] + VL[prs, 0])
    FL[vxn, 0] = UL[vxn, 0]*VL[vxn, 0]
    if UL.shape[0] > 3:
        FL[vxt, 0] = UL[vxn, 0]*VL[vxt, 0]
    if UL.shape[0] > 4:
        FL[vxb, 0] = UL[vxn, 0]*VL[vxb, 0]

    FR[rho, 0] = UR[vxn, 0]
    FR[eng, 0] = VR[vxn, 0]*(UR[eng, 0] + VR[prs, 0])
    FR[vxn, 0] = UR[vxn, 0]*VR[vxn, 0]
    if UR.shape[0] > 3:
        FR[vxt, 0] = UR[vxn, 0]*VR[vxt, 0]
    if UR.shape[0] > 4:
        FR[vxb, 0] = UR[vxn, 0]*VR[vxb, 0]

    # Riemann solver (tvdlf)
    VLR = np.empty(shape=VL.shape, dtype=np.float64)
    VLR[0, 0] = 0.5*(VL[0, 0] + VR[0, 0])
    VLR[vxn, 0] = 0.5*(np.absolute(VL[vxn, 0]) + np.absolute(VR[vxn, 0]))

    csLR = np.empty(shape=VLR.shape[1], dtype=np.float64)
    csLR[0] = np.sqrt(gamma*VLR[prs, 0]/VLR[rho, 0])

    Smax = np.maximum(np.absolute(VLR[vxn, 0] + csLR[0]),
                      np.absolute(VLR[vxn, 0] - csLR[0]))

    flux = np.empty(shape=FL.shape, dtype=np.float64)
    pres = np.empty(shape=FL.shape, dtype=np.float64)
    flux = 0.5*(FL[0, 0] + FR[0, 0] - Smax*(UR[0, 0] - UL[0, 0]))
    pres = 0.5*(VL[prs, 0] + VR[prs, 0])

    if Smax.max() > speed_max:
        speed_max = Smax.max()

    dflux = np.empty(shape=(flux.shape[0], flux.shape[0]-1), dtype=np.float64)
    dflux = -(flux[0, 1] - flux[0, -1])*dtdx
    dflux[vxn, 0] -= (pres[1] - pres[-1])*dtdx

    return dflux, speed_max


#@nb.stencil
def TwoD_first_order_stencil(U, gamma, gamma_1, igamma_1, speed_max, dtdx, vxn, vxt, vxb):

    V = np.empty(shape=U.shape, dtype=np.float64)

    # Conservitive to primative
    V[rho, :] = U[rho, :]

    irho = np.empty(shape=U.shape[1], dtype=np.float64)
    m2 = np.empty(shape=U.shape[1], dtype=np.float64)

    irho[:] = 1.0/V[rho, :]

    m2[:] = U[mvx1, :]**2
    V[vx1, :] = U[mvx1, :]*irho[:]
    if U.shape[0] > 3:
        m2[:] += U[mvx2, :]**2
        V[vx2, :] = U[mvx2, :]*irho[:]
    if U.shape[0] > 4:
        m2[:] += U[mvx3, :]**2
        V[vx3, :] = U[mvx3, :]*irho[:]

    V[prs, :] = gamma_1*(U[eng, :] - 0.5*m2*irho[:])

    # Reconstruction
    VL, VR = V[:, :-1], V[:, 1:]

    UL = np.empty(shape=VL.shape, dtype=np.float64)
    UR = np.empty(shape=VR.shape, dtype=np.float64)

    # Primative to conservative
    v2 = np.empty(shape=UL.shape[1], dtype=np.float64)
    UL[rho, :] = VL[rho, :]
    v2[:] = VL[vx1, :]**2
    UL[mvx1, :] = VL[vx1, :]*VL[rho, :]
    if UL.shape[0] > 3:
        v2 += VL[vx2, :]**2
        UL[mvx2, :] = VL[vx2, :]*VL[rho, :]
    if UL.shape[0] > 4:
        v2 += VL[vx3, :]**2
        UL[mvx3, :] = VL[vx3, :]*VL[rho, :]
    UL[eng, :] = 0.5*VL[rho, :]*v2 + VL[prs, :]*igamma_1

    UR[rho, :] = VR[rho, :]
    v2 = VR[vx1, :]**2
    UR[mvx1, :] = VR[vx1, :]*VR[rho, :]
    if UR.shape[0] > 3:
        v2 += VR[vx2, :]**2
        UR[mvx2, :] = VR[vx2, :]*VR[rho, :]
    if UR.shape[0] > 4:
        v2 += VR[vx3, :]**2
        UR[mvx3, :] = VR[vx3, :]*VR[rho, :]
    UR[eng, :] = 0.5*VR[rho, :]*v2 + VR[prs, :]*igamma_1

    FL = np.empty(shape=VL.shape, dtype=np.float64)
    FR = np.empty(shape=VR.shape, dtype=np.float64)

    # Flux tensor
    FL[rho, :] = UL[vxn, :]
    FL[eng, :] = VL[vxn, :]*(UL[eng, :] + VL[prs, :])
    FL[vxn, :] = UL[vxn, :]*VL[vxn, :]
    if UL.shape[0] > 3:
        FL[vxt, :] = UL[vxn, :]*VL[vxt, :]
    if UL.shape[0] > 4:
        FL[vxb, :] = UL[vxn, :]*VL[vxb, :]

    FR[rho, :] = UR[vxn, :]
    FR[eng, :] = VR[vxn, :]*(UR[eng, :] + VR[prs, :])
    FR[vxn, :] = UR[vxn, :]*VR[vxn, :]
    if UR.shape[0] > 3:
        FR[vxt, :] = UR[vxn, :]*VR[vxt, :]
    if UR.shape[0] > 4:
        FR[vxb, :] = UR[vxn, :]*VR[vxb, :]

    # Riemann solver (tvdlf)
    VLR = np.empty(shape=VL.shape, dtype=np.float64)
    VLR[:, :] = 0.5*(VL[:, :] + VR[:, :])
    VLR[vxn, :] = 0.5*(np.absolute(VL[vxn, :]) + np.absolute(VR[vxn, :]))

    csLR = np.empty(shape=VLR.shape[1], dtype=np.float64)
    csLR[:] = np.sqrt(gamma*VLR[prs, :]/VLR[rho, :])

    Smax = np.maximum(np.absolute(VLR[vxn, :] + csLR[:]),
                      np.absolute(VLR[vxn, :] - csLR[:]))

    flux = np.empty(shape=FL.shape, dtype=np.float64)
    pres = np.empty(shape=FL.shape, dtype=np.float64)
    flux = 0.5*(FL[:, :] + FR[:, :] - Smax*(UR[:, :] - UL[:, :]))
    pres = 0.5*(VL[prs, :] + VR[prs, :])

    if Smax.max() > speed_max:
        speed_max = Smax.max()

    dflux = np.empty(shape=(flux.shape[0], flux.shape[0]-1), dtype=np.float64)
    dflux = -(flux[:, 1:] - flux[:, :-1])*dtdx
    dflux[vxn, :] -= (pres[1:] - pres[:-1])*dtdx

    return dflux, speed_max
