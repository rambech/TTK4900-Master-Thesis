import numpy as np
import casadi as ca
from utils import D2R, Smtrx, Hmtrx, m2c, Rzyx, sat, crossFlowDrag
from .models import Model


class OtterModel(Model):
    def __init__(self, dt: float = 0.05, N: int = 40) -> None:
        super().__init__(dt, N)
        self._init_model()

    def _init_opt(self, x_init, u_init, opti: ca.Opti):
        # Declaring optimization variables
        # State variables
        x = opti.variable(6, self.N+1)

        # Input variables
        u = opti.variable(2, self.N)

        # Slack variables
        s = opti.variable(6, self.N+1)

        # Control signal and time constraint
        opti.subject_to(opti.bounded(-1, u[0, :], 1))
        opti.subject_to(opti.bounded(-1, u[1, :], 1))

        # Boundary values
        # Initial conditions
        opti.subject_to(x[0, 0] == x_init[0])
        opti.subject_to(x[1, 0] == x_init[1])
        opti.subject_to(x[2, 0] == x_init[2])
        opti.subject_to(x[3, 0] == x_init[3])
        opti.subject_to(x[4, 0] == x_init[4])
        opti.subject_to(x[5, 0] == x_init[5])
        opti.subject_to(u[0, 0] == u_init[0])
        opti.subject_to(u[1, 0] == u_init[1])

        # Initial guesses
        opti.set_initial(x[0, :], x_init[0])
        opti.set_initial(x[1, :], x_init[1])
        opti.set_initial(x[2, :], x_init[2])
        opti.set_initial(x[3, :], x_init[3])
        opti.set_initial(x[4, :], x_init[4])
        opti.set_initial(x[5, :], x_init[5])
        opti.set_initial(u[0, :], u_init[0])
        opti.set_initial(u[1, :], u_init[1])

        return x, u, s

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

    def step(self, eta: np.ndarray, nu: np.ndarray, prev_u: np.ndarray,
             action: np.ndarray, beta_c: float, V_c: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Step method
        [nu,u_feedback] = step(eta,nu,u_feedback,action,beta_c,V_c) integrates
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

    def _denormalise(self, action):
        u = np.zeros(2).astype(np.float32)
        for idx, a in enumerate(action):
            if a < 0:
                u[idx] = a*111
            else:
                u[idx] = a*113

        return u

    def single_shooting(self, x_init, u_init, opti: ca.Opti):
        # Declaring optimization variables
        # State variables
        x = opti.variable(3, self.N+1)
        x_pos = x[0, :]
        y_pos = x[1, :]
        theta = x[2, :]

        # Input variables
        u = opti.variable(2, self.N)
        v = u[0, :]
        phi = u[1, :]

        # Slack variables
        s = opti.variable(3, self.N+1)

        # Fixed step Runge-Kutta 4 integrator
        self.RK4(x, u, self.N, opti)

        # TODO: Put this code in a separate method
        # ----------------------------------------------------
        # Control signal and time constraint
        opti.subject_to(opti.bounded(-1, v, 2))
        opti.subject_to(opti.bounded(D2R(-15), phi, D2R(15)))
        # ----------------------------------------------------

        # Boundary values
        # Initial conditions
        opti.subject_to(x_pos[0] == x_init[0])
        opti.subject_to(y_pos[0] == x_init[1])
        opti.subject_to(theta[0] == x_init[2])
        opti.subject_to(v[0] == u_init[0])
        opti.subject_to(phi[0] == u_init[1])

        # Initial guesses
        opti.set_initial(x_pos, x_init[0])
        opti.set_initial(y_pos, x_init[1])
        opti.set_initial(theta, x_init[2])
        opti.set_initial(v, u_init[0])
        opti.set_initial(phi, u_init[1])

        return x, u, s
