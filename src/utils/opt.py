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


def R(psi: float) -> np.ndarray:
    """
    Simple 2x2 rotation matrix

    Parameters
    ----------
        psi : float
            Heading angle

    Returns
    -------
        R : np.ndarray
            2D rotation matrix

    """

    row1 = ca.horzcat(ca.cos(psi), -ca.sin(psi))
    row2 = ca.horzcat(ca.sin(psi), ca.cos(psi))

    return ca.vertcat(row1, row2)

# ------------------------------------------------------------------------------


def Rz(psi: float) -> ca.MX:
    """
    Rotation matrix around the z axis

    Parameters
    ----------
        psi : float
            Heading angle

    Returns
    -------
        Rz : ca.MX
            2D rotation matrix

    """

    row1 = ca.horzcat(ca.cos(psi), -ca.sin(psi), 0)
    row2 = ca.horzcat(ca.sin(psi), ca.cos(psi), 0)
    row3 = ca.horzcat(0, 0, 1)

    return ca.vertcat(row1, row2, row3)

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


# ------------------------------------------------------------------------------


def simple_quadratic(x: ca.DM, x_d: ca.DM,
                     config: dict = None, slack: ca.DM = None):
    """
    Simple quadratic objective function

    min (x_t - x_d)^2 + (y_t - y_d)^2 + (psi_t - psi_d)^2

    Parameters
    ----------
        x : ca.Opti.variable
            State variables
        x_d : ca.Opti.parameter
            Desired end state
        config : dict
            Penalty weights for objective function
        slack : ca.Opti.variable
            Slack variable

    Returns
    -------
        L : ca.DM
            Objective function
    """

    L = (
        config["Q"][0][0]*(x[0, -1]-x_d[0, -1])**2 +
        config["Q"][1][1]*(x[1, -1]-x_d[1, -1])**2 +
        config["Q"][2][2]*(x[2, -1]-x_d[2, -1])**2
    )

    if slack is not None:
        print("Using slack")
        L += (
            config["q_slack"][0]*slack[0]**2 +
            config["q_slack"][1]*slack[1]**2 +
            config["q_slack"][2]*slack[2]**2
        )

        if slack.shape[0] > 3:
            L += (
                config["q_slack"][3]*slack[3]**2 +
                config["q_slack"][4]*slack[4]**2 +
                config["q_slack"][5]*slack[5]**2
            )

    return L

# ------------------------------------------------------------------------------


def _pos_linear(x, x_d, delta):
    """
    Position eucliean distance cost

    f_xy(eta_N, eta_d) = delta * sqrt((x - x_d)**2 + (y - y_d)**2))

    """

    return delta * ca.sqrt((x[0] - x_d[0])**2 + (x[1] - x_d[1])**2)

# ------------------------------------------------------------------------------


def _pos_pseudo_huber(x, x_d, delta):
    """
    Position pseudo-Huber cost

    f_xy(eta_N, eta_d) = delta**2 * (sqrt(1 + ((x - x_d)**2 + (y - y_d)**2) / delta**2) - 1)

    """

    return delta**2 * (ca.sqrt(1 + ((x[0] - x_d[0])**2 +
                                    (x[1] - x_d[1])**2) / delta**2) - 1)

# ------------------------------------------------------------------------------


def _heading_cost(x, x_d):
    """
    Heading reward

    f_psi(eta_N, eta_d) = (1 - cos(psi - psi_d))/2

    """

    # print(f"x_d[2]: {x_d[2]}")

    return (1 - ca.cos(x[2] - x_d[2]))/2

# ------------------------------------------------------------------------------


def pseudo_huber(x, u, x_d, config: dict = None, slack=None):
    """
    Full objective function utilizing pseudo-Huber

    min q_xy * f_xy(eta_N, eta_d) + q_psi * f_psi(eta_N, eta_d)
        + sum(nu.T.dot(Q.dot(nu)) + tau.T.dot(R.dot(tau)))

    Parameters
    ----------
        x : Any
            6 x 1, state decision variable

    """

    L = (
        config["q_xy"]*_pos_pseudo_huber(x, x_d, config["delta"]) +
        config["q_psi"]*_heading_cost(x, x_d) +
        x[3:6].T @ np.asarray(config["Q"]) @ x[3:6] +
        u.T @ np.asarray(config["R"]) @ u
    )

    if slack is not None:
        L += (
            ca.MX(config["q_slack"]).T @ slack
        )

    return L


def linear_quadratic(x, u, x_d, config: dict = None, slack=None):
    """
    Full objective function utilizing euclidean distance

    min q_xy * f_xy(eta_N, eta_d) + q_psi * f_psi(eta_N, eta_d)
        + sum(nu.T.dot(Q.dot(nu)) + tau.T.dot(R.dot(tau)))

    Parameters
    ----------
        x : Any
            6 x 1, state decision variable

    """

    L = (
        config["q_xy"]*_pos_linear(x, x_d, config["delta"]) +
        config["q_psi"]*_heading_cost(x, x_d) +
        x[3:6].T @ np.asarray(config["Q"]) @ x[3:6] +
        u.T @ np.asarray(config["R"]) @ u
    )

    if slack is not None:
        L += (
            ca.MX(config["q_slack"]).T @ slack
        )

    return L
