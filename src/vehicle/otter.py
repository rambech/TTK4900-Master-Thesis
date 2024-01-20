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

# TODO: Add particle bursts from each thruster


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

        # self.nu = np.zeros(6, float)    # velocity vector
        # self.eta = np.zeros(6, float)   # position vector

        # propeller revolution states
        # self.u_feedback = np.zeros(2, float)
        self.name = "Otter USV (see 'otter.py' for more details)"

        self._init_model()

        # For render
        self.scale = 30  # [px/m]
        # self.vessel_image = pygame.Surface(
        #     (self.scale*self.L, self.scale*self.B))
        # self.vessel_image.fill((239, 129, 20))  # NTNU Orange
        self.vessel_image = pygame.image.load(
            'vehicle/images/otter.png')
        self.vessel_image = pygame.transform.scale(
            self.vessel_image, (self.scale*self.B, self.scale*self.L))

    def _init_model(self):
        # Constants
        self.D2R = np.pi / 180     # deg2rad
        self.g = 9.81              # acceleration of gravity (m/s^2)
        rho = 1026                 # density of water (kg/m^3)

        # Initialize the Otter USV model
        self.T_n = 1.0  # propeller time constants (s)
        self.L = 2.0    # Length (m)
        self.B = 1.08   # beam (m)

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
        Yv = 0
        Zw = -2 * 0.3 * w3 * self.M[2, 2]  # specified using relative damping
        Kp = -2 * 0.2 * w4 * self.M[3, 3]
        Mq = -2 * 0.4 * w5 * self.M[4, 4]
        Nr = -self.M[5, 5] / T_yaw  # specified by the time constant T_yaw

        self.D = -np.diag([Xu, Yv, Zw, Kp, Mq, Nr])

        # Propeller configuration/input matrix
        B = self.k_pos * np.array([[1, 1], [-self.l1, -self.l2]])
        self.Binv = np.linalg.inv(B)

    def step(self, eta, nu, prev_u, tau_d, beta_c, V_c):
        """
        Normal step method for simulation
        """
        u_control = self.unconstrained_allocation(tau_d)

        # Normalise to make it go up
        u_control = self._normalise(u_control)

        nu, u = self.rl_step(
            eta, nu, prev_u, u_control, beta_c, V_c)

        return nu, u

    def rl_step(self, eta, nu, prev_u, action, beta_c, V_c) -> tuple[np.ndarray, np.ndarray]:
        """
        Step method for RL purposes
        [nu,u_feedback] = rl_step(eta,nu,u_feedback,action,beta_c,V_c) integrates
        the Otter USV equations of motion using Euler's method.
        """
        # Denormalise from rl
        action = self._denormalise(action)

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
        CA[5, 0] = 0  # assume that the Munk moment in yaw can be neglected
        CA[5, 1] = 0  # if nonzero, must be balanced by adding nonlinear damping
        CA[0, 5] = 0
        CA[1, 5] = 0

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
            n[i] = sat(n[i], self.n_min, self.n_max)

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
        tau_damp = -np.matmul(self.D, nu_r)
        tau_damp[5] = tau_damp[5] - 10 * self.D[5, 5] * abs(nu_r[5]) * nu_r[5]

        # State derivatives (with dimension)
        tau_crossflow = crossFlowDrag(self.L, self.B_pont, self.T, nu_r)
        sum_tau = (
            tau
            + tau_damp
            + tau_crossflow
            - np.matmul(C, nu_r)
            - np.matmul(self.G, eta)
            # + g_0
        )

        # np.matmul(self.Minv, sum_tau)  # USV dynamics
        nu_dot = self.Minv.dot(sum_tau)
        n_dot = (action - n) / self.T_n  # propeller dynamics

        # Forward Euler integration [k+1]
        nu = nu + self.dt * nu_dot
        n = n + self.dt * n_dot

        u = np.array(n, float)

        return nu, u

    def render(self, eta: np.ndarray, offset: tuple[float, float]):
        rotated_image = pygame.transform.rotate(
            self.vessel_image, -R2D(eta[-1])).convert_alpha()
        eta_s = N2S(eta, self.scale, offset)
        center = (eta_s[0], eta_s[1])
        rect = rotated_image.get_rect(center=center)

        return rotated_image, rect

    def unconstrained_allocation(self, tau) -> np.ndarray:
        u_control = self.Binv @ tau

        return u_control

    def corners(self, eta: np.ndarray) -> list:
        """
        Returns the corners of the vehicle given its position

        x_v^n = l * cos(psi + delta)
        y_v^n = l * sin(psi + delta)

        where psi is heading and delta is the angle between x_b
        and the corners

        Parameters
        ----------
            eta : np.ndarray
                Pose in {n}

        Returns
        -------
            corners : np.ndarray
                Outer corners of the vessel given in (x,y) in {n}
        """

        half_L = self.L/2
        half_B = self.B/2
        l = np.linalg.norm((half_L, half_B), 2)
        psi_1 = eta[-1] + np.arctan2(half_B, half_L)
        psi_2 = eta[-1] + np.arctan2(half_B, -half_L)
        psi_3 = eta[-1] + np.arctan2(-half_B, -half_L)
        psi_4 = eta[-1] + np.arctan2(-half_B, half_L)

        forward_starboard = (eta[0]+l*np.cos(psi_1), eta[1]+l*np.sin(psi_1))
        forward_port = (eta[0]+l*np.cos(psi_2), eta[1]+l*np.sin(psi_2))
        aft_port = (eta[0]+l*np.cos(psi_3), eta[1]+l*np.sin(psi_3))
        aft_starboard = (eta[0]+l*np.cos(psi_4), eta[1]+l*np.sin(psi_4))

        corners = [forward_starboard, forward_port, aft_port, aft_starboard]

        return corners

    def _normalise(self, u):
        action = np.zeros(2).astype(np.float32)
        for idx, n in enumerate(u):
            if n < 0:
                action[idx] = (n/111)
            else:
                action[idx] = (n/113)

        return action

    def _denormalise(self, action):
        u = np.zeros(2).astype(np.float32)
        for idx, a in enumerate(action):
            if a < 0:
                u[idx] = a*111
            else:
                u[idx] = a*113

        return u
