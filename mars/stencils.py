import numpy as np
import numba as nb
import sys

from settings import *
from tools import flux_tensor, cons_to_prims, prims_to_cons
from riemann_solvers import tvdlf, hll, hllc
from reconstruction import flat, minmod

#@nb.jit(cache=True)
def OneD_first_order_stencil(U, gamma, gamma_1, igamma_1, speed_max, dtdx, vxn, vxt, vxb):

    V = np.zeros(shape=U.shape, dtype=np.float64)

    if np.isnan(U[rho]).any() or  np.isnan(V[rho]).any():
        print("rho is nan before:", U[rho], V[rho])
        sys.exit()

    # Conservitive to primative
    #print("before", U[rho], V[rho])
    cons_to_prims(U, V, gamma_1)

    #print("After", U[rho], V[rho])
    #sys.exit()
    if np.isnan(U[rho]).any() or  np.isnan(V[rho]).any():
        print("rho is nan after:", U[rho], V[rho])
        sys.exit()

    # if np.isnan(V[rho]).any():
    #     print("V[rho] is nan:", V[rho])
    #     sys.exit()

    if (V[rho] < 0).any():
        print("U[rho] < 0:", U[rho])
        sys.exit()
    #print("Stencil V shape:", V.shape)

    # Reconstruction
    VL, VR = minmod(V)

    #print("Left and right states for V after reconstruction:", VL.shape, VR.shape)

    UL = np.zeros(shape=VL.shape, dtype=np.float64)
    UR = np.zeros(shape=VR.shape, dtype=np.float64)

    # Primative to conservative
    prims_to_cons(VL, UL, igamma_1)
    prims_to_cons(VR, UR, igamma_1)

    FL = np.zeros(shape=VL.shape, dtype=np.float64)
    FR = np.zeros(shape=VR.shape, dtype=np.float64)

    # Flux tensor
    flux_tensor(UL, VL, FL, vxn, vxt, vxb)
    flux_tensor(UR, VR, FR, vxn, vxt, vxb)

    # Riemann solver (tvdlf)
    dflux, speed_max = tvdlf(
        FL, FR, UL, UR, VL, VR, speed_max, gamma, dtdx, vxn, vxt, vxb
    )
    if np.isnan(dflux).any():
        print("dflux is nan:", dflux[rho], U[rho], V[rho])
        sys.exit()
    return dflux, speed_max
