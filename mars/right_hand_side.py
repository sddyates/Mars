
import sys
import numpy as np
from settings import *
from tools import flux_tensor, cons_to_prims, prims_to_cons, time_step


def face_flux_RK2(U, g, a, vxn, vxt, vxb):
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

    V = np.zeros(shape=U.shape, dtype=np.float64)
    cons_to_prims(U, V, a.gamma_1)
    del U

    g.VL, g.VR = a.reconstruction(V, g.gz, g.dxi[vxn-2])

    prims_to_cons(g.VL, g.UL, a.gamma_1)
    prims_to_cons(g.VR, g.UR, a.gamma_1)

    flux_tensor(g.UL, g.VL, g.FL, vxn, vxt, vxb)
    flux_tensor(g.UR, g.VR, g.FR, vxn, vxt, vxb)

    g.flux, g.pres, g.cs_max, g.speed_max = a.riemann_solver(g.flux, g.FL, g.FR, g.UL, g.UR, g.VL, g.VR, g.SL, g.SR, g.pres, g.cs_max, g.speed_max, a.gamma, vxn, vxt, vxb)

    #if np.isnan(np.sum(g.flux)):
    #    print("Error, nan in array, function: riemann")
    #    sys.exit()

    return


def face_flux_MH(U, g, a, vxn, vxt, vxb):
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

    ic = g.gz + 1
    jc = g.gz + 1
    kc = g.gz + 1
    stencil = 2*g.gz + 1

    g.build_fluxes(vxn)

    array_shape = U.shape

    g.flux = np.zeros(shape=(array_shape[0], 2))
    g.FL = np.zeros(shape=(array_shape[0], 2))
    g.FR = np.zeros(shape=(array_shape[0], 2))
    g.UL = np.zeros(shape=(array_shape[0], 2))
    g.UR = np.zeros(shape=(array_shape[0], 2))
    g.VL = np.zeros(shape=(array_shape[0], 2))
    g.VR = np.zeros(shape=(array_shape[0], 2))
    g.SL = np.zeros(shape=stencil)
    g.SR = np.zeros(shape=stencil)
    g.pres = np.zeros(shape=stencil)
    g.cmax = np.zeros(shape=stencil)

    V = cons_to_prims(U, a)

    V_x1m = np.zeros(shape=(array_shape[0], 2))
    V_x1p = np.zeros(shape=(array_shape[0], 2))

    V_x2m = np.zeros(shape=(array_shape[0], 2))
    V_x2p = np.zeros(shape=(array_shape[0], 2))

    V_x3m = np.zeros(shape=(array_shape[0], 2))
    V_x3p = np.zeros(shape=(array_shape[0], 2))

    V_x1p, V_x1m = a.reconstruction(V[:, kc, jc, :], g.gz, g.dxi[0])
    V_x2p, V_x2m = a.reconstruction(V[:, kc, :, ic], g.gz, g.dxi[1])
    V_x3p, V_x3m = a.reconstruction(V[:, :, jc, ic], g.gz, g.dxi[2])

    V_Hatx = V_x \
        - 0.5*dt/g.dx1*JAx1(V[:, kc, jc, ic], a)*(V_x1p[:, 1] - Vx1m[:, 0]) \
        - 0.5*dt/g.dx2*JAx2(V[:, kc, jc, ic], a)*(V_x2p[:, 1] - Vx2m[:, 0]) \
        - 0.5*dt/g.dx3*JAx3(V[:, kc, jc, ic], a)*(V_x3p[:, 1] - Vx3m[:, 0])

    U_x1m = prims_to_cons(V_x1m, a)
    U_x1p = prims_to_cons(V_x1p, a)
    U_x2m = prims_to_cons(V_x2m, a)
    U_x2p = prims_to_cons(V_x2p, a)
    U_x3m = prims_to_cons(V_x3m, a)
    U_x3p = prims_to_cons(V_x3p, a)

    g.FL = flux_tensor(g.UL, a, vxn, vxt, vxb)
    g.FR = flux_tensor(g.UR, a, vxn, vxt, vxb)
    a.riemann_solver(g, a, vxn, vxt, vxb)
    dflux_x1 = g.flux[:, 0] - g.flux[:, 1]
    dflux_x1[mvx1] -= g.pres[1] - g.pres[0]

    a.riemann_solver(g, a, vxn, vxt, vxb)
    dflux_x1 = g.flux[:, 0] - g.flux[:, 1]
    dflux_x1[mvx1] -= g.pres[1] - g.pres[0]

    a.riemann_solver(g, a, vxn, vxt, vxb)
    dflux_x1 = g.flux[:, 0] - g.flux[:, 1]
    dflux_x1[mvx1] -= g.pres[1] - g.pres[0]

    F_ip = Flux_MH(U_x1p[:, 1], U_x1m[:, 1])
    F_im = Flux_MH(U_x1p[:, 0], U_x1p[:, 0])

    F_jp = Flux_MH(U_x2p[:, 1], U_y2m[:, 1])
    F_jm = Flux_MH(U_x2m[:, 0], U_x2p[:, 0])

    F_kp = Flux_MH(U_x3p[:, 1], U_x3m[:, 1])
    F_km = Flux_MH(U_x3m[:, 0], U_x3p[:, 0])

    a.riemann_solver(g, a, vxn, vxt, vxb)

    if np.isnan(np.sum(g.flux)):
        print("Error, nan in array, function: riemann")
        sys.exit()

    return


def RHSOperator(U, g, a, dt):
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

        face_flux_RK2(U, g, a, vxn=2, vxt=3, vxb=4)
        dflux_x1[:, g.ibeg:g.iend] = -(g.flux[:, 1:] - g.flux[:, :-1])
        dflux_x1[mvx1, g.ibeg:g.iend] -= g.pres[1:] - g.pres[:-1]

        dflux = dflux_x1/g.dx1
        del dflux_x1

    if a.is_2D:

        dflux_x1 = np.zeros(shape=U.shape)
        dflux_x2 = np.zeros(shape=U.shape)

        for j in range(g.jbeg, g.jend):
            face_flux_RK2(U[:, j, :], g, a, vxn=2, vxt=3, vxb=4)
            dflux_x1[:, j, g.ibeg:g.iend] = -(g.flux[:, 1:] - g.flux[:, :-1])
            dflux_x1[mvx1, j, g.ibeg:g.iend] -= g.pres[1:] - g.pres[:-1]

        for i in range(g.ibeg, g.iend):
            face_flux_RK2(U[:, :, i], g, a, vxn=3, vxt=2, vxb=4)
            dflux_x2[:, g.jbeg:g.jend, i] = -(g.flux[:, 1:] - g.flux[:, :-1])
            dflux_x2[mvx2, g.jbeg:g.jend, i] -= g.pres[1:] - g.pres[:-1]

        dflux = dflux_x1/g.dx1 + dflux_x2/g.dx2
        del dflux_x1, dflux_x2

    if a.is_3D:

        dflux_x1 = np.zeros(shape=U.shape)
        dflux_x2 = np.zeros(shape=U.shape)
        dflux_x3 = np.zeros(shape=U.shape)

        for k in range(g.kbeg, g.kend):
            for j in range(g.jbeg, g.jend):
                face_flux_RK2(U[:, k, j, :], g, a, vxn=2, vxt=3, vxb=4)
                dflux_x1[:, k, j, g.ibeg:g.iend] = -(g.flux[:, 1:] - g.flux[:, :-1])
                dflux_x1[mvx1, k, j, g.ibeg:g.iend] -= g.pres[1:] - g.pres[:-1]

        for k in range(g.kbeg, g.kend):
            for i in range(g.ibeg, g.iend):
                face_flux_RK2(U[:, k, :, i], g, a, vxn=3, vxt=2, vxb=4)
                dflux_x2[:, k, g.jbeg:g.jend, i] = -(g.flux[:, 1:] - g.flux[:, :-1])
                dflux_x2[mvx2, k, g.jbeg:g.jend, i] -= g.pres[1:] - g.pres[:-1]

        for j in range(g.jbeg, g.jend):
            for i in range(g.ibeg, g.iend):
                face_flux_RK2(U[:, :, j, i], g, a, vxn=4, vxt=2, vxb=3)
                dflux_x3[:, g.kbeg:g.kend, j, i] = -(g.flux[:, 1:] - g.flux[:, :-1])
                dflux_x3[mvx3, g.kbeg:g.kend, j, i] -= g.pres[1:] - g.pres[:-1]

        dflux = dflux_x1/g.dx1 + dflux_x2/g.dx2 + dflux_x3/g.dx3
        del dflux_x1, dflux_x2, dflux_x3

    #if np.isnan(np.sum(dflux)):
    #    print("Error, nan in array, function: flux")
    #    sys.exit()

    return dt*dflux


def RHSOperator_MH(U, g, a, dt):
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

        for i in range(g.ibeg, g.iend):
            face_flux_MH(U, g, a, vxn=2, vxt=3, vxb=4)
            dflux_x1[:, g.ibeg:g.iend] = -(g.flux[:, 1:] - g.flux[:, :-1])
            dflux_x1[mvx1, g.ibeg:g.iend] -= g.pres[1:] - g.pres[:-1]

        dflux = dt*dflux_x1/g.dx1

    if a.is_2D:

        dflux_x1 = np.zeros(shape=U.shape)
        dflux_x2 = np.zeros(shape=U.shape)

        for j in range(g.jbeg, g.jend):
            face_flux_MH(U[:, j, :], g, a, vxn=2, vxt=3, vxb=4)
            dflux_x1[:, j, g.ibeg:g.iend] = -(g.flux[:, 1:] - g.flux[:, :-1])
            dflux_x1[mvx1, j, g.ibeg:g.iend] -= g.pres[1:] - g.pres[:-1]

        for i in range(g.ibeg, g.iend):
            face_flux_MH(U[:, :, i], g, a, vxn=3, vxt=2, vxb=4)
            dflux_x2[:, g.jbeg:g.jend, i] = -(g.flux[:, 1:] - g.flux[:, :-1])
            dflux_x2[mvx2, g.jbeg:g.jend, i] -= g.pres[1:] - g.pres[:-1]

        dflux = dflux_x1 + dflux_x2

    if a.is_3D:

        dflux_x1 = np.zeros(shape=U.shape)
        dflux_x2 = np.zeros(shape=U.shape)
        dflux_x3 = np.zeros(shape=U.shape)

        for k in range(g.kbeg, g.kend):
            for j in range(g.jbeg, g.jend):
                for i in range(g.ibeg, g.iend):

                    face_flux_MH(U[:, k-g.gz:k+g.gz, j-g.gz:j+g.gz, i-g.gz:i+g.gz], g, a, vxn=2, vxt=3, vxb=4)
                    dflux_x1[:, k, j, i] = g.flux[:, :-1] - g.flux[:, 1:]
                    dflux_x1[mvx1, k, j, i] -= g.pres[1:] - g.pres[:-1]

                    face_flux_MH(U[:, k, j-g.gz:j+g.gz, i], g, a, vxn=3, vxt=2, vxb=4)
                    dflux_x2[:, k, j, i] = g.flux[:, :-1] - g.flux[:, 1:]
                    dflux_x2[mvx2, k, j, i] -= g.pres[1:] - g.pres[:-1]

                    face_flux_MH(U[:, k-g.gz:k+g.gz, j, i], g, a, vxn=4, vxt=2, vxb=3)
                    dflux_x3[:, k, j, i] = g.flux[:, :-1] - g.flux[:, 1:]
                    dflux_x3[mvx3, k, j, i] -= g.pres[1:] - g.pres[:-1]

        dflux = dflux_x1/g.dx1 + dflux_x2/g.dx2 + dflux_x3/g.dx3

    if np.isnan(np.sum(dflux)):
        print("Error, nan in array, function: flux")
        sys.exit()

    return dflux
