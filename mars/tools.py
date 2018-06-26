import numpy as np
import sys
from settings import *


def eigenvalues(UL, UR, p, axis):

    VL = cons_to_prims(UL, p)
    VR = cons_to_prims(UR, p)

    VLR = 0.5*(VL + VR)

    csL = np.sqrt(p['gamma']*VL[prs]/VL[rho])
    csR = np.sqrt(p['gamma']*VR[prs]/VR[rho])

    if(np.isnan(csL).any() or np.isnan(csR).any()):
        print("Erroar, nan found in eigenvalues:")
        print('csL=', csL)
        print('csR=', csR)
        print('VL[rho]=', VL[rho])
        print('VL[prs]=', VL[prs])
        print('VR[rho]=', VR[rho])
        print('VR[prs]=', VR[prs])
        sys.exit()

    if axis == 'i':
        SL = np.minimum(VL[vx1] - csL, VR[vx1] - csR)
        SR = np.maximum(VL[vx1] + csL, VR[vx1] + csR)
    elif axis == 'j':
        SL = np.minimum(VL[vx2] - csL, VR[vx2] - csR)
        SR = np.maximum(VL[vx2] + csL, VR[vx2] + csR)
    elif axis == 'k':
        SL = np.minimum(VL[vx3] - csL, VR[vx3] - csR)
        SR = np.maximum(VL[vx3] + csL, VR[vx3] + csR)

    return SL, SR


def cons_to_prims(U, p):

    V = np.zeros(shape=U.shape)

    m2 = U[mvx1]*U[mvx1]
    if p['Dimensions'] == '2D':
        m2 += U[mvx2]*U[mvx2]
    if p['Dimensions'] == '3D':
        m2 += U[mvx2]*U[mvx2]
        m2 += U[mvx3]*U[mvx3]

    kinE = 0.5*m2/U[rho]

    if (U[eng, U[eng, :] < 0.0] < 0.0).any():
        U[eng, U[eng, :] < 0.0] = small_pressure/(p['gamma'] - 1.0) \
            + kinE[U[eng, :] < 0.0]

    V[rho] = U[rho]
    V[vx1] = U[mvx1]/U[rho]
    if p['Dimensions'] == '2D':
        V[vx2] = U[mvx2]/U[rho]
    if p['Dimensions'] == '3D':
        V[vx2] = U[mvx2]/U[rho]
        V[vx3] = U[mvx3]/U[rho]
    V[prs] = (p['gamma'] - 1.0)*(U[eng] - kinE)

    if (V[prs, V[prs, :] < 0.0] < 0.0).any():
        V[prs, V[prs, :] < 0.0] = small_pressure

    if np.isnan(V).any():
        print("Error, nan in cons_to_prims")
        sys.exit()

    return V


def prims_to_cons(V, p):

    U = np.zeros(shape=V.shape)

    v2 = V[vx1]*V[vx1]
    if p['Dimensions'] == '2D':
        v2 += V[vx2]*V[vx2]
    if p['Dimensions'] == '3D':
        v2 += V[vx2]*V[vx2]
        v2 += V[vx3]*V[vx3]

    U[rho] = V[rho]
    U[mvx1] = V[rho]*V[vx1]
    if p['Dimensions'] == '2D':
        U[mvx2] = V[rho]*V[vx2]
    if p['Dimensions'] == '3D':
        U[mvx2] = V[rho]*V[vx2]
        U[mvx3] = V[rho]*V[vx3]
    U[eng] = 0.5*V[rho]*v2 + V[prs]/(p['gamma'] - 1.0)

    if np.isnan(U).any():
        print("Error, nan in prims_to_cons")
        sys.exit()

    return U 


def time_step(V, g, p):

    if p['Dimensions'] == '1D':

        cs = np.sqrt(p['gamma']\
            *V[prs, g.ibeg:g.iend]\
            /V[rho, g.ibeg:g.iend])

        max_velocity = np.amax(abs(V[vx1, g.ibeg:g.iend]))
        max_speed = np.amax(abs(V[vx1, g.ibeg:g.iend]) + cs)
        dt = p['cfl']*g.dx1/max_speed 
        mach_number = np.amax(abs(V[vx1, g.ibeg:g.iend])/cs)

    elif p['Dimensions'] == '2D':

        cs = np.sqrt(p['gamma']\
            *V[prs, g.jbeg:g.jend, g.ibeg:g.iend]\
            /V[rho, g.jbeg:g.jend, g.ibeg:g.iend])

        max_velocity = np.amax(abs(V[vx1:vx2, g.jbeg:g.jend, g.ibeg:g.iend]))
        max_speed = np.amax(abs(V[vx1:vx2, g.jbeg:g.jend, g.ibeg:g.iend]) + cs)
        dt = p['cfl']*min(g.dx1, g.dx2)/max_speed 
        mach_number = np.amax(abs(V[vx1:vx2, g.jbeg:g.jend, g.ibeg:g.iend])/cs)

    elif p['Dimensions'] == '3D':

        cs = np.sqrt(p['gamma']\
            *V[prs, g.kbeg:g.kend, g.jbeg:g.jend, g.ibeg:g.iend]\
            /V[rho, g.kbeg:g.kend, g.jbeg:g.jend, g.ibeg:g.iend])

        max_velocity = np.amax(
            abs(V[vx1:vx3, g.kbeg:g.kend, g.jbeg:g.jend, g.ibeg:g.iend]))
        max_speed = np.amax(
            abs(V[vx1:vx3, g.kbeg:g.kend, g.jbeg:g.jend, g.ibeg:g.iend]) + cs)
        dt = p['cfl']*min(g.dx1, g.dx2, g.dx3)/max_speed 
        mach_number = np.amax(
            abs(V[vx1:vx3, g.kbeg:g.kend, g.jbeg:g.jend, g.ibeg:g.iend])/cs)

    if np.isnan(dt):
        print("Error, nan in time_step, cs =", cs)
        sys.exit()

    return dt, max_velocity, mach_number


