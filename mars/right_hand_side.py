
import sys
import numpy as np
from settings import *
from tools import flux_tensor, cons_to_prims, prims_to_cons, time_step


def face_flux(U, g, a, vxn, vxt, vxb):
    """
    Synopsis
    --------
    Construct the fluxes through the cell 
    faces normal to the direction of "axis".  

    Args
    ----
    U: numpy array-like
    state vector containing all 
    conservative variables.

    g: object-like
    object containing all variables related to 
    the grid, e.g. cell width.

    a: object-like
    object containing specified algorithms for use 
    in the seprate stages of a time step.       

    vx(n,t,b): int-like
    indexes for for the normal, tangential and 
    bi-tangential velocity components.

    Attributes
    ----------
    None.

    TODO
    ----
    None
    """

    g.build_fluxes(vxn)

    V = cons_to_prims(U, a)

    g.VL, g.VR = a.reconstruction(V, g.gz, g.dxi[vxn-2])

    g.UL = prims_to_cons(g.VL, a)
    g.UR = prims_to_cons(g.VR, a)

    g.FL = flux_tensor(g.UL, a, vxn, vxt, vxb)
    g.FR = flux_tensor(g.UR, a, vxn, vxt, vxb)

    a.riemann_solver(g, a, vxn, vxt, vxb)

    if np.isnan(np.sum(g.flux)):
        print("Error, nan in array, function: riemann")
        sys.exit()

    return


def RHSOperator(U, g, a):
    """
    Synopsis
    --------
    Determine the right hand side operator.

    Args
    ----
    U: numpy array-like
    state vector containing all 
    conservative variables.

    g: object-like
    object containing all variables related to 
    the grid, e.g. cell width.

    a: object-like
    object containing specified algorithms for use 
    in the seprate stages of a time step.       

    Attributes
    ----------
    None.

    TODO
    ----
    None
    """

    #for axis in g.axis_list:

    #    for axis in range(beg, end):
    #        face_flux(U[:, :, :, :], g, a, p, axis)
    #        dflux_x1[:, k, j, g.ibeg:g.iend] = -(g.flux[:, 1:] - g.flux[:, :-1])
    #        dflux_x1[mvx1, k, j, g.ibeg:g.iend] -= g.pres[1:] - g.pres[:-1]

    #    geometry.swich_to_j_from_i()

    if a.is_1D:

        dflux_x1 = np.zeros(shape=U.shape)

        face_flux(U, g, a, vxn=2, vxt=3, vxb=4)
        dflux_x1[:, g.ibeg:g.iend] = -(g.flux[:, 1:] - g.flux[:, :-1])
        dflux_x1[mvx1, g.ibeg:g.iend] -= g.pres[1:] - g.pres[:-1]

        dflux = dflux_x1/g.dx1

    if a.is_2D:

        dflux_x1 = np.zeros(shape=U.shape)
        dflux_x2 = np.zeros(shape=U.shape)

        for j in range(g.jbeg, g.jend):
            face_flux(U[:, j, :], g, a, vxn=2, vxt=3, vxb=4)
            dflux_x1[:, j, g.ibeg:g.iend] = -(g.flux[:, 1:] - g.flux[:, :-1])
            dflux_x1[mvx1, j, g.ibeg:g.iend] -= g.pres[1:] - g.pres[:-1]

        for i in range(g.ibeg, g.iend):
            face_flux(U[:, :, i], g, a, vxn=3, vxt=2, vxb=4)
            dflux_x2[:, g.jbeg:g.jend, i] = -(g.flux[:, 1:] - g.flux[:, :-1])
            dflux_x2[mvx2, g.jbeg:g.jend, i] -= g.pres[1:] - g.pres[:-1]

        dflux = dflux_x1/g.dx1 + dflux_x2/g.dx2

    if a.is_3D:

        dflux_x1 = np.zeros(shape=U.shape)
        dflux_x2 = np.zeros(shape=U.shape)
        dflux_x3 = np.zeros(shape=U.shape)

        for k in range(g.kbeg, g.kend):
            for j in range(g.jbeg, g.jend):
                face_flux(U[:, k, j, :], g, a, vxn=2, vxt=3, vxb=4)
                dflux_x1[:, k, j, g.ibeg:g.iend] = -(g.flux[:, 1:] - g.flux[:, :-1])
                dflux_x1[mvx1, k, j, g.ibeg:g.iend] -= g.pres[1:] - g.pres[:-1]

        for k in range(g.kbeg, g.kend):
            for i in range(g.ibeg, g.iend):
                face_flux(U[:, k, :, i], g, a, vxn=3, vxt=2, vxb=4)
                dflux_x2[:, k, g.jbeg:g.jend, i] = -(g.flux[:, 1:] - g.flux[:, :-1])
                dflux_x1[mvx2, k, g.jbeg:g.jend, i] -= g.pres[1:] - g.pres[:-1]

        for j in range(g.jbeg, g.jend): 
            for i in range(g.ibeg, g.iend):
                face_flux(U[:, :, j, i], g, a, vxn=4, vxt=2, vxb=3)
                dflux_x3[:, g.kbeg:g.kend, j, i] = -(g.flux[:, 1:] - g.flux[:, :-1])
                dflux_x1[mvx3, g.kbeg:g.kend, j, i] -= g.pres[1:] - g.pres[:-1]

        dflux = dflux_x1/g.dx1 + dflux_x2/g.dx2 + dflux_x3/g.dx3

    if np.isnan(np.sum(dflux)):
        print("Error, nan in array, function: flux")
        sys.exit()

    return dflux
