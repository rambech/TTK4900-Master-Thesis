import numpy as np
from utils import Smtrx, Hmtrx
from casadi import vertcat, sin, cos


class Model():
    """
    Base class for vehicle models
    """

    def __init__(self) -> None:
        pass

    def _init_model(self) -> None:
        ...


class DubinsCarModel(Model):
    def __init__(self) -> None:
        pass

    def step(self, x, u):
        """
        MPC model step for DubinsCarModel

        Parameters
        ----------
        x : np.ndarray
            State vector, x = [x, y, theta]
        u : np.ndarray
            Control input, u[0] = v and u[1] = phi
        """
        return vertcat(u[0]*cos(x[2]),
                       u[0]*sin(x[2]),
                       u[0]*u[1])


class OtterModel(Model):
    def __init__(self) -> None:
        self._init_model()

    def _init_model(self):
        # Constants
        self.g = 9.81              # acceleration of gravity (m/s^2)
        rho = 1026                 # density of water (kg/m^3)

        # Initialize the Otter USV model
        self.T_n = 1.0  # Propeller time constants (s)
        self.L = 2.0    # Length (m)
        self.B = 1.08   # Beam (m)
        self.dof = 2    # Number of DOFs

        self.controls = [
            "Left propeller shaft speed (rad/s)",
            "Right propeller shaft speed (rad/s)"
        ]
        self.dimU = len(self.controls)

        # Vehicle parameters
        m = 55.0                                 # mass (kg)
        self.mp = 25.0                           # Payload (kg)
        self.m_total = m + self.mp
        self.rp = np.array([0.05, 0, -0.35], float)  # location of payload (m)
        rg = np.array([0.2, 0, -0.2], float)     # CG for hull only (m)
        rg = (m * rg + self.mp * self.rp) / \
            (m + self.mp)  # CG corrected for payload
        self.S_rg = Smtrx(rg)
        self.H_rg = Hmtrx(rg)
        self.S_rp = Smtrx(self.rp)

        R44 = 0.4 * self.B  # radii of gyration (m)
        R55 = 0.25 * self.L
        R66 = 0.25 * self.L
        T_yaw = 1.0         # time constant in yaw (s)
        Umax = 6 * 0.5144   # max forward speed (m/s)

        # Data for one pontoon
        self.B_pont = 0.25  # beam of one pontoon (m)
        # distance from centerline to waterline centroid (m)
        y_pont = 0.395
        Cw_pont = 0.75      # waterline area coefficient (-)
        Cb_pont = 0.4       # block coefficient, computed from m = 55 kg

        # Inertia dyadic, volume displacement and draft
        nabla = (m + self.mp) / rho  # volume
        self.T = nabla / (2 * Cb_pont * self.B_pont * self.L)  # draft
        Ig_CG = m * np.diag(np.array([R44 ** 2, R55 ** 2, R66 ** 2]))
        self.Ig = Ig_CG - m * self.S_rg @ self.S_rg - self.mp * self.S_rp @ self.S_rp

        # Experimental propeller data including lever arms
        self.l1 = -y_pont  # lever arm, left propeller (m)
        self.l2 = y_pont  # lever arm, right propeller (m)
        self.k_pos = 0.02216 / 2  # Positive Bollard, one propeller
        self.k_neg = 0.01289 / 2  # Negative Bollard, one propeller
        # max. prop. rev.
        self.n_max = np.sqrt((0.5 * 24.4 * self.g) / self.k_pos)
        # min. prop. rev.
        self.n_min = -np.sqrt((0.5 * 13.6 * self.g) / self.k_neg)

        # MRB_CG = [ (m+mp) * I2  O2      (Fossen 2021, Chapter 3)
        #               O2        Ig ]
        MRB_CG = np.zeros((3, 3))
        MRB_CG[0:2, 0:2] = (m + self.mp) * np.eye(2)
        MRB_CG[2:3, 2:3] = self.Ig
        MRB = self.H_rg.T @ MRB_CG @ self.H_rg

        # Hydrodynamic added mass (best practice)
        Xudot = -0.1 * m
        Yvdot = -1.5 * m
        # Zwdot = -1.0 * m
        # Kpdot = -0.2 * self.Ig[0, 0]
        # Mqdot = -0.8 * self.Ig[1, 1]
        Nrdot = -1.7 * self.Ig[2, 2]

        self.MA = -np.diag([Xudot, Yvdot, Nrdot])

        # System mass matrix
        self.M = MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Hydrostatic quantities (Fossen 2021, Chapter 4)
        Aw_pont = Cw_pont * self.L * self.B_pont  # waterline area, one pontoon
        I_T = (
            2
            * (1 / 12)
            * self.L
            * self.B_pont ** 3
            * (6 * Cw_pont ** 3 / ((1 + Cw_pont) * (1 + 2 * Cw_pont)))
            + 2 * Aw_pont * y_pont ** 2
        )
        I_L = 0.8 * 2 * (1 / 12) * self.B_pont * self.L ** 3
        KB = (1 / 3) * (5 * self.T / 2 - 0.5 * nabla / (self.L * self.B_pont))
        BM_T = I_T / nabla  # BM values
        BM_L = I_L / nabla
        KM_T = KB + BM_T    # KM values
        KM_L = KB + BM_L
        KG = self.T - rg[2]
        GM_T = KM_T - KG    # GM values
        GM_L = KM_L - KG

        G33 = rho * self.g * (2 * Aw_pont)  # spring stiffness
        G44 = rho * self.g * nabla * GM_T
        G55 = rho * self.g * nabla * GM_L
        G_CF = np.diag([0, 0, G33, G44, G55, 0])  # spring stiff. matrix in CF
        LCF = -0.2
        H = Hmtrx(np.array([LCF, 0.0, 0.0]))  # transform G_CF from CF to CO
        self.G = H.T @ G_CF @ H

        # Natural frequencies
        w3 = np.sqrt(G33 / self.M[2, 2])
        w4 = np.sqrt(G44 / self.M[3, 3])
        w5 = np.sqrt(G55 / self.M[4, 4])

        # Linear damping terms (hydrodynamic derivatives)
        Xu = -24.4 * self. g / Umax  # specified using the maximum speed
        Yv = 0
        # Zw = -2 * 0.3 * w3 * self.M[2, 2]  # specified using relative damping
        Kp = -2 * 0.2 * w4 * self.M[3, 3]
        Mq = -2 * 0.4 * w5 * self.M[4, 4]
        Nr = -self.M[5, 5] / T_yaw  # specified by the time constant T_yaw

        self.D = -np.diag([Xu, Yv, Kp, Mq, Nr])

        # Propeller configuration/input matrix
        B = self.k_pos * np.array([[1, 1], [-self.l1, -self.l2]])
        self.Binv = np.linalg.inv(B)

    def step(self, x, u):
        """
        Step Otter NMPC model

        Parameters:
        x : np.ndarray
            State vector, x = [x, y, psi, u, v, r]
        u : 

        """
        ...
