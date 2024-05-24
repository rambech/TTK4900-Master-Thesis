import numpy as np
import casadi as ca
import pygame
from gymnasium import spaces

import utils
from .vehicle import Vehicle


class SimpleOtter(Vehicle):
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
        m = 62  # mass (kg)
        self.mp = 0                           # Payload (kg)
        self.m_total = m + self.mp
        self.rp = np.array([0.05, 0, -0.35], float)  # location of payload (m)
        rg = np.array([0.2, 0, -0.2], float)     # CG for hull only (m)
        rg = (m * rg + self.mp * self.rp) / \
            (m + self.mp)  # CG corrected for payload
        self.xg = rg[0]
        self.S_rg = utils.Smtrx(rg)

        # Use mask to remove terms containing
        # heave, roll and pitch
        mask = np.array([True, True, False, False, False, True])
        self.H_rg = utils.Hmtrx(rg)[mask][:, mask]

        self.S_rp = utils.Smtrx(self.rp)

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
        # Cw_pont = 0.75      # waterline area coefficient (-)
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
        # self.k_neg = 0.01289 / 2  # Negative Bollard, one propeller
        self.k_port = self.k_pos
        self.k_stb = self.k_pos

        # MRB = [m   0    0      (Fossen 2021, Chapter 6)
        #        0   m   mxg
        #        0  mxg   Iz]
        MRB = np.zeros((3, 3))
        MRB[0, 0] = self.m_total
        MRB[1, 1] = self.m_total
        MRB[1, 2] = self.m_total * self.xg
        MRB[2, 1] = MRB[1, 2]
        MRB[2, 2] = self.Ig[-1, -1]

        # Hydrodynamic added mass (best practice)
        Xudot = -0.1 * m
        Yvdot = -1.5 * m
        Nrdot = -1.7 * self.Ig[2, 2]

        self.MA = -np.diag([Xudot, Yvdot, Nrdot])

        # System mass matrix
        self.M = MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Linear damping terms (hydrodynamic derivatives)
        Xu = -24.4 * self. g / Umax  # specified using the maximum speed
        # specified using the time constant in sway
        Yv = -self.M[1, 1] / T_sway
        Nr = -self.M[-1, -1] / T_yaw  # specified by the time constant T_yaw

        # Linear damping
        self.D = -np.diag([Xu, Yv, Nr])

        # Nonlinear damping
        self.Nrr = 10 * Nr

        # Propeller configuration/input matrix
        # TODO: This is probably wrong
        B = np.array([[self.k_port, 0],
                      [0, self.k_stb]]).dot(np.array([[1, 1], [-self.l1, -self.l2]]))
        self.Binv = np.linalg.inv(B)

        self.theta = np.array(
            [
                self.m_total, self.Ig[-1, -1], self.xg,
                Xudot, Yvdot, Nrdot, Xu, Yv, Nr, self.Nrr,
                self.k_port, self.k_stb
            ]
        )

    def step(self, eta: np.ndarray, nu: np.ndarray, prev_u: np.ndarray,
             u: np.ndarray, beta_c: float, V_c: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Step method
        [nu,u_feedback] = step(eta,nu,u_feedback,action,beta_c,V_c) integrates
        the Otter USV equations of motion using Euler's method.

        Parameters
        -----------
            x : ca.Opti.variable
                State space containing pose and velocity in 3-DOF
            u : np.ndarray
                Current control input

        Returns
        -------
            xdot : ca.Opti.variable
                Derivative of eta and nu


        """

        # =======================
        # Prep decision variables
        # =======================
        # Break nu down to 3-DOF
        small_nu = np.array([nu[0], nu[1], nu[5]])

        # Input vector
        n = [u[0], u[1]]

        # Current velocities
        u_c = V_c * np.cos(beta_c - eta[5])  # current surge vel.
        v_c = V_c * np.sin(beta_c - eta[5])  # current sway vel.

        # Current velocity vector
        nu_c = np.array([u_c, v_c, 0], float)
        nu_r = small_nu - nu_c  # relative velocity vector
        # ===============
        # Coriolis matrix
        # ===============
        # CRB based on assumptions from
        # Fossen 2021, Chapter 6, page 137
        CRB = np.zeros((3, 3))
        CRB[0, 1] = -self.m_total * nu_r[2]
        CRB[0, 2] = -self.m_total * self.xg * nu_r[2]
        CRB[1, 0] = -CRB[0, 1]
        CRB[2, 0] = -CRB[0, 2]

        # Added coriolis with Munk moment
        CA = utils.m2c(self.MA, nu_r)
        C = CRB + CA

        # ======================
        # Thrust dynamics
        # ======================
        thrust = np.array(
            [
                self.k_port * n[0]*np.sqrt(n[0]**2),
                self.k_stb * n[1]*np.sqrt(n[1]**2)
            ]
        )

        # Control forces and moments
        tau = ca.vertcat(thrust[0] + thrust[1],
                         0,
                         -self.l1 * thrust[0] - self.l2 * thrust[1])
        tau = np.array(
            [
                thrust[0] + thrust[1],
                0,
                -self.l1 * thrust[0] - self.l2 * thrust[1]
            ]
        )

        # ================
        # Calculate forces
        # ================
        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = self.D @ nu_r
        tau_damp[2] = tau_damp[2] - self.Nrr * np.sqrt(nu_r[2]**2) * nu_r[2]

        # =========================
        # Solve the Fossen equation
        # =========================
        sum_tau = (
            tau
            - tau_damp
            - C @ nu_r
        )

        # ==================
        # Calculate dynamics
        # ==================
        nu_dot = self.Minv @ sum_tau
        # Pad nu_dot to fit simulator
        nu_dot = np.array([nu_dot[0], nu_dot[1], 0, 0, 0, nu_dot[2]])

        nu = nu + self.dt * nu_dot

        return nu, u
