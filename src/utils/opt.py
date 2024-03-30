"""
Optimization utilities 

Author: @rambech
"""

import numpy as np
import casadi as ca
import utils

# ------------------------------------------------------------------------------


def Smtrx(x):
    """
    Skew-symmetric matrix S(x) = -S(x).T

    Parameters
    ----------
        x : Any
            Vector with 3 elements

    Returns
    -------
        S : ca.matrix
            Skew-symmetric matrix

    """


# ------------------------------------------------------------------------------


def m2c(M: np.ndarray, nu: ca.DM) -> ca.DM:
    """
    C = m2c(M,nu) computes the Coriolis and centripetal matrix C from the
    mass matrix M and generalized velocity vector nu (Fossen 2021, Ch. 3)

    3 DOF Casadi version

    Parameters
    ----------
        M : np.ndarray
            Added? mass matrix, shape(3,3)
        nu : ca.DM
            Velocity vector nu = [u, v, r].T
    """

    # NOTE: Yrdot, i.e. M(1,2) is zero for the Otter model
    M = 0.5 * (M + M.T)     # systematization of the inertia matrix
    # 3-DOF model (surge, sway and yaw)
    # C = [         0             0      M(1,1)*v+M(1,2)*r
    #               0             0         -M(0,0)*u
    #      -M(1,1)*v-M(1,2)*r  M(0,0)*u         0          ]
    c02 = M[1, 1]*nu[1] + M[1, 2]*nu[2]
    c12 = -M[0, 0]*nu[0]
    c20 = -c02
    c21 = -c12
    surge_col = ca.vertcat(0,
                           0,
                           c20)
    sway_col = ca.vertcat(0,
                          0,
                          c21)
    yaw_col = ca.vertcat(c02,
                         c12,
                         0)
    coriolis_matrix = ca.horzcat(surge_col, sway_col, yaw_col)

    return coriolis_matrix


# ------------------------------------------------------------------------------


def simple_m2c(M: np.ndarray, nu: ca.DM) -> ca.DM:
    """
    C = simple_m2c(M,nu) computes the Coriolis and centripetal matrix C from the
    mass matrix M and generalized velocity vector nu, without Munk moment 
    (Fossen 2021, Ch. 3)

    3 DOF Casadi version

    Parameters
    ----------
        M : np.ndarray
            Added? mass matrix, shape(3,3)
        nu : ca.DM
            Velocity vector nu = [u, v, r].T
    """

    # NOTE: Yrdot, i.e. M(1,2) is zero for the Otter model
    M = 0.5 * (M + M.T)     # systematization of the inertia matrix
    # 3-DOF model (surge, sway and yaw)
    # C = [         0             0      M(1,1)*v+M(1,2)*r
    #               0             0         -M(0,0)*u
    #      -M(1,1)*v-M(1,2)*r  M(0,0)*u         0          ]
    c02 = M[1, 1]*nu[1] + M[1, 2]*nu[2]
    c12 = -M[0, 0]*nu[0]
    c20 = -c02
    c21 = -c12
    surge_col = ca.vertcat(0,
                           0,
                           c20)
    sway_col = ca.vertcat(0,
                          0,
                          c21)
    yaw_col = ca.vertcat(c02,
                         c12,
                         0)
    coriolis_matrix = ca.horzcat(surge_col, sway_col, yaw_col)

    return coriolis_matrix


# ------------------------------------------------------------------------------


def Rz(psi: float) -> np.ndarray:
    """
    Rotation matrix around the z axis

    Parameters
    ----------
        psi : float
            Heading angle

    Returns
    -------
        Rz : np.ndarray
            2D rotation matrix

    """

    col1 = ca.vertcat(ca.cos(psi),
                      ca.sin(psi),
                      0)
    col2 = ca.vertcat(-ca.sin(psi),
                      ca.cos(psi),
                      0)
    col3 = ca.vertcat(0,
                      0,
                      1)

    return ca.horzcat(col1, col2, col3)

# ------------------------------------------------------------------------------


def Rzyx(phi, theta, psi):
    """
    R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
    using the zyx convention
    """

    cphi = ca.cos(phi)
    sphi = ca.sin(phi)
    cth = ca.cos(theta)
    sth = ca.sin(theta)
    cpsi = ca.cos(psi)
    spsi = ca.sin(psi)

    col1 = ca.vertcat(cpsi*cth,
                      spsi*cth,
                      -sth)
    col2 = ca.vertcat(-spsi*cphi+cpsi*sth*sphi,
                      cpsi*cphi+sphi*sth*spsi,
                      cth*sphi)
    col3 = ca.vertcat(spsi*sphi+cpsi*cphi*sth,
                      -cpsi*sphi+sth*spsi*cphi,
                      cth*cphi)

    R = ca.horzcat(col1, col2, col3)

    return R

# ------------------------------------------------------------------------------


def crossFlowDrag(L, B, T, nu_r):
    """
    tau_crossflow = crossFlowDrag(L,B,T,nu_r) computes the cross-flow drag 
    integrals for a marine craft using strip theory. 

    M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_crossflow
    """

    rho = 1026               # density of water
    n = 20                   # number of strips

    dx = L/20
    # 2D drag coefficient based on Hoerner's curve
    Cd_2D = utils.Hoerner(B, T)

    Yh = 0
    Nh = 0
    xL = -L/2

    for i in range(0, n+1):
        v_r = nu_r[1]             # relative sway velocity
        r = nu_r[-1]              # yaw rate
        Ucf = ca.fabs(v_r + xL * r) * (v_r + xL * r)
        Yh = Yh - 0.5 * rho * T * Cd_2D * Ucf * dx         # sway force
        Nh = Nh - 0.5 * rho * T * Cd_2D * xL * Ucf * dx    # yaw moment
        xL += dx

    tau_crossflow = ca.vertcat(0,
                               Yh,
                               Nh)

    return tau_crossflow
