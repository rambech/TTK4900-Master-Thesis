"""
Otter vehicle model
This is Thor Inge Fossen's code adapted for 
pygame and stable baselines.
Original: 
https://github.com/cybergalactic/PythonVehicleSimulator/blob/master/src/python_vehicle_simulator/vehicles/otter.py


Important to remember:
Rendering coordinates are defined form the top left corner of the simulation
screen, while eta is defined from the center of the screen
"""

from gymnasium import spaces
import numpy as np
import pygame

from .vehicle import Vehicle
from utils import Smtrx, Hmtrx, Rzyx, m2c, crossFlowDrag, sat, R2D, N2S


class Otter(Vehicle):
    """
    Otter vehicle class
    """

    def __init__(self, seed=None, dt=0.02) -> None:
        self.dt = dt
        self.u = np.zeros(2, float)

        # Otter parameters for observation space
        psi_max = 2*np.pi   # [rad] Heading
        psi_min = -psi_max  # [rad]
        # [m/s] Linear velocity in x (surge) direction, body frame
        u_max = 3.08667
        u_min = -3.08667    # [m/s]
        # [m/s] Linear velocity in y (sway) direction, body frame
        v_max = u_max
        # [m/s] Just a guess, have no clue of what to put here
        v_min = -v_max
        r_max = psi_max*dt  # [rad/s] Heading rate
        r_min = psi_min*dt  # [rad/s]

        nu_max = np.array([u_max, v_max, r_max])
        nu_min = np.array([u_min, v_min, r_min])

        self.info = {"Name": "Otter"}
        self.limits = {"psi_max": 2*np.pi,
                       "psi_min": -2*np.pi,
                       "nu_max": nu_max,
                       "nu_min": nu_min,
                       "u_max": [113.0, 113.0],
                       "u_min": [-111.0, -111.0]}

        # Action space for the otter is +-100%
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), seed=seed)

        self.name = "Otter USV (see 'otter.py' for more details)"

        self._init_model()

        # For render
        self.vessel_image = pygame.image.load(
            'vehicle/images/newotter.png'
        )

    def _init_model(self):
        # Constants
        self.g = 9.81   # acceleration of gravity (m/s^2)
        rho = 1026      # density of water (kg/m^3)

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
        m = 62  # 55.0                                 # mass (kg)
        self.mp = 0  # 25.0                           # Payload (kg)
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
        T_sway = 1.0        # time constant in sway (s)
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

        # MRB_CG = [ (m+mp) * I3  O3      (Fossen 2021, Chapter 3)
        #               O3        Ig ]
        MRB_CG = np.zeros((6, 6))
        MRB_CG[0:3, 0:3] = (m + self.mp) * np.eye(3)
        MRB_CG[3:6, 3:6] = self.Ig
        MRB = self.H_rg.T @ MRB_CG @ self.H_rg

        # Hydrodynamic added mass (best practice)
        Xudot = -0.1 * m
        Yvdot = -1.5 * m
        Zwdot = -1.0 * m
        Kpdot = -0.2 * self.Ig[0, 0]
        Mqdot = -0.8 * self.Ig[1, 1]
        Nrdot = -1.7 * self.Ig[2, 2]

        self.MA = -np.diag([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot])

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
        # specified using the time constant in sway
        Yv = -self.M[1, 1] / T_sway
        Zw = -2 * 0.3 * w3 * self.M[2, 2]  # specified using relative damping
        Kp = -2 * 0.2 * w4 * self.M[3, 3]
        Mq = -2 * 0.4 * w5 * self.M[4, 4]
        Nr = -self.M[5, 5] / T_yaw  # specified by the time constant T_yaw

        self.D = -np.diag([Xu, Yv, Zw, Kp, Mq, Nr])

        # Propeller configuration/input matrix
        B = self.k_pos * np.array([[1, 1], [-self.l1, -self.l2]])
        self.Binv = np.linalg.inv(B)

        self.theta = np.array(
            [
                self.m_total, self.Ig[-1, -1], rg[0]*self.m_total,
                Xudot, Yvdot, Nrdot, Xu, Yv, Nr, 10 * Nr,
                self.k_pos, self.k_pos
            ]
        )

    def step(self, eta: np.ndarray, nu: np.ndarray, prev_u: np.ndarray,
             action: np.ndarray, beta_c: float, V_c: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Step method
        [nu,u_feedback] = step(eta,nu,u_feedback,action,beta_c,V_c) integrates
        the Otter USV equations of motion using Euler's method.
        """
        # Denormalise from rl
        # action = self._denormalise(action)

        # Input vector
        n = np.array([prev_u[0], prev_u[1]])

        # Current velocities
        u_c = V_c * np.cos(beta_c - eta[5])  # current surge vel.
        v_c = V_c * np.sin(beta_c - eta[5])  # current sway vel.

        # current velocity vector
        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)
        nu_r = nu - nu_c  # relative velocity vector

        # Rigid body and added mass Coriolis and centripetal matrices
        # CRB_CG = [ (m+mp) * Smtrx(nu2)          O3           (Fossen 2021, Chapter 6)
        #              O3                   -Smtrx(Ig*nu2)  ]
        CRB_CG = np.zeros((6, 6))
        CRB_CG[0:3, 0:3] = self.m_total * Smtrx(nu[3:6])
        CRB_CG[3:6, 3:6] = -Smtrx(np.matmul(self.Ig, nu[3:6]))
        CRB = self.H_rg.T @ CRB_CG @ self.H_rg  # transform CRB from CG to CO

        CA = m2c(self.MA, nu_r)
        # CA[5, 0] = 0  # assume that the Munk moment in yaw can be neglected
        # CA[5, 1] = 0  # if nonzero, must be balanced by adding nonlinear damping
        # CA[0, 5] = 0
        # CA[1, 5] = 0

        C = CRB + CA

        # Payload force and moment expressed in BODY
        R = Rzyx(eta[3], eta[4], eta[5])
        f_payload = np.matmul(R.T, np.array([0, 0, self.mp * self.g], float))
        m_payload = np.matmul(self.S_rp, f_payload)
        g_0 = np.array([f_payload[0], f_payload[1], f_payload[2],
                        m_payload[0], m_payload[1], m_payload[2]])

        # Control forces and moments - with propeller revolution saturation
        thrust = np.zeros(2)
        for i in range(0, 2):

            # saturation, physical limits
            n[i] = sat(self.n_min, n[i], self.n_max)

            if n[i] > 0:  # positive thrust
                thrust[i] = self.k_pos * n[i] * abs(n[i])
            else:  # negative thrust
                thrust[i] = self.k_neg * n[i] * abs(n[i])

        # Control forces and moments
        tau = np.array(
            [
                thrust[0] + thrust[1],
                0,
                0,
                0,
                0,
                -self.l1 * thrust[0] - self.l2 * thrust[1],
            ]
        )

        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = -np.matmul(self.D, nu_r)  # neg
        tau_damp[5] = tau_damp[5] - 10 * \
            self.D[5, 5] * abs(nu_r[5]) * nu_r[5]  # neg

        # State derivatives (with dimension)
        tau_crossflow = crossFlowDrag(self.L, self.B_pont, self.T, nu_r)
        sum_tau = (
            tau
            + tau_damp  # neg neg
            + tau_crossflow
            - np.matmul(C, nu_r)
            - np.matmul(self.G, eta)
            + g_0
        )

        # np.matmul(self.Minv, sum_tau)  # USV dynamics
        nu_dot = self.Minv.dot(sum_tau)
        n_dot = (action - n) / self.T_n  # propeller dynamics

        # Forward Euler integration [k+1]
        nu = nu + self.dt * nu_dot
        n = n + self.dt * n_dot

        u = np.array(n, float)

        return nu, u
