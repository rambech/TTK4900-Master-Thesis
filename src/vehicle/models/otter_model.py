import numpy as np
import casadi as ca
import utils
from .models import Model


class OtterModel(Model):
    def __init__(self, dt: float = 0.05, N: int = 40, rl: bool = False) -> None:
        super().__init__(dt, N, rl)
        self._init_model()

    def _init_opt(self, x_init, u_init, opti: ca.Opti):
        # Declaring optimization variables
        # State variables
        x = opti.variable(6, self.N+1)
        north = x[0, :]
        east = x[1, :]
        yaw = x[2, :]
        surge = x[3, :]
        sway = x[4, :]
        yaw_rate = x[5, :]

        # Input variables
        u = opti.variable(2, self.N)
        port_u = u[0, :]
        starboard_u = u[1, :]

        # Slack variables
        s = opti.variable(3, self.N+1)

        # Control signal and time constraint
        opti.subject_to(opti.bounded(-100, port_u, 100))
        opti.subject_to(opti.bounded(-100, starboard_u, 100))

        # Boundary values
        # Initial state conditions
        opti.subject_to(north[0] == x_init[0])
        opti.subject_to(east[0] == x_init[1])
        opti.subject_to(yaw[0] == x_init[2])
        opti.subject_to(surge[0] == x_init[3])
        opti.subject_to(sway[0] == x_init[4])
        opti.subject_to(yaw_rate[0] == x_init[5])

        # Initial control input conditions
        opti.subject_to(port_u[0] == u_init[0])
        opti.subject_to(starboard_u[0] == u_init[1])

        # # Slack constraints
        opti.subject_to(s[0] <= 0)
        opti.subject_to(s[1] <= 0)
        opti.subject_to(s[2] <= 0)

        # Initial guesses for state variables
        opti.set_initial(north, x_init[0])
        opti.set_initial(east, x_init[1])
        opti.set_initial(yaw, x_init[2])
        opti.set_initial(surge, x_init[3])
        opti.set_initial(sway, x_init[4])
        opti.set_initial(yaw_rate, x_init[5])

        # Initila guesses for control inputs
        opti.set_initial(port_u, u_init[0])
        opti.set_initial(starboard_u, u_init[1])

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
        self.k_starboard = self.k_pos

        # # max. prop. rev.
        # self.n_max = np.sqrt((0.5 * 24.4 * self.g) / self.k_pos)
        # # min. prop. rev.
        # self.n_min = -np.sqrt((0.5 * 13.6 * self.g) / self.k_neg)

        # MRB_CG = [ (m+mp) * I2  O2      (Fossen 2021, Chapter 3)
        #               O2        Ig ]
        # MRB_CG = np.zeros((3, 3))
        # MRB_CG[0:2, 0:2] = (m + self.mp) * np.eye(2)

        # For 3-DOF, Ig = Iz
        # MRB_CG[2:3, 2:3] = self.Ig[-1, -1]
        # MRB = self.H_rg.T @ MRB_CG @ self.H_rg

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

        self.D = -np.diag([Xu, Yv, Nr])

        # Propeller configuration/input matrix
        B = np.array([[self.k_port, 0],
                      [0, self.k_starboard]]).dot(np.array([[1, 1], [-self.l1, -self.l2]]))
        Binv = np.linalg.inv(B)
        self.Binv = np.array([Binv[0],
                              [0, 0],
                              Binv[1]])

        # Cross-sectional weights see Martinsen
        self.W = np.array([self.B, self.L, 1])

        # Environment forces in NED
        self.w = np.zeros(3)

    def update_model(self, model_params: dict) -> None:
        """
        Update model parameters
        Takes in model parameters from another place and updates

        Parameters
        -----------
            model_params : dict

        Returns
        -------
            self
        """

        # Update mass and damping coefficients
        self.M = model_params["M_matrix"]
        self.Minv = np.linalg.inv(self.M)

        self.D = -np.diag(model_params["D_vector"])

        self.w = model_params["w"]

        # Update thruster coefficients
        self.k_port = model_params["k_port"]
        self.k_starboard = model_params["k_starboard"]

        # We go left to right port first then starboard
        B = np.array([[self.k_port, 0],
                      [0, self.k_starboard]]).dot(np.array([[1, 1], [-self.l1, -self.l2]]))
        Binv = np.linalg.inv(B)

        # NOTE: self.Binv is wrong!
        self.Binv = np.array([Binv[0],
                              [0, 0,],
                              Binv[1]])

        # TODO: Add update to n_min and n_max as well

    def step(self, x: ca.Opti.variable, u: ca.Opti.variable, prev_u: ca.Opti.variable = None) -> tuple[ca.Opti.variable, ca.Opti.variable]:
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
            nu : ca.Opti.variable
                Updated velocity
            u : ca.Opti.variable
                Updated control input

        """

        # =======================
        # Prep decision variables
        # =======================
        # Split states into eta and nu
        eta = x[:3]
        nu = x[3:]

        # Input vector
        if prev_u is not None:
            n = ca.vertcat(prev_u[0],
                           prev_u[1])
        else:
            n = ca.vertcat(u[0],
                           u[1])

        # ===============
        # Coriolis matrix
        # ===============
        if self.rl:
            # TODO: Determine if this is correct
            C = utils.opt.m2c(self.M, nu)
        else:
            # CRB based on assumptions from
            # Fossen 2021, Chapter 6, page 137
            CRB = ca.MX.zeros(3, 3)
            CRB[0, 1] = -self.m_total * nu[2]
            CRB[0, 2] = -self.m_total * self.xg * nu[2]
            CRB[1, 0] = -CRB[0, 1]
            CRB[2, 0] = -CRB[0, 2]

            # Added coriolis with Munk moment
            CA = utils.opt.m2c(self.MA, nu)
            C = CRB + CA

        # ======================
        # Thrust dynamics
        # ======================
        # thrust = n
        thrust = ca.vertcat(self.k_port * n[0]*ca.fabs(n[0]),
                            self.k_starboard * n[1]*ca.fabs(n[1]))

        # Control forces and moments
        tau = ca.vertcat(thrust[0] + thrust[1],
                         0,
                         -self.l1 * thrust[0] - self.l2 * thrust[1])

        # ================
        # Calculate forces
        # ================
        # tau = self.Binv @ thrust
        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = -self.D @ nu
        tau_damp[-1] = tau_damp[-1] - 10 * \
            self.D[-1, -1] * ca.fabs(nu[-1]) * nu[-1]

        # =========================
        # Solve the Fossen equation
        # =========================
        if self.rl:
            sum_tau = (
                tau
                + tau_damp
                + self.W @ utils.opt.Rz(eta[2]).T @ self.w
                - C @ nu
            )
        else:
            sum_tau = (
                tau
                + tau_damp
                - C @ nu
            )

        # ==================
        # Calculate dynamics
        # ==================
        # Transform nu from {b} to {n}
        eta_dot = utils.opt.Rz(eta[2]) @ nu
        nu_dot = self.Minv @ sum_tau

        # Construct state vector
        x_dot = ca.vertcat(eta_dot,
                           nu_dot)

        if prev_u is not None:
            n_dot = (u - n) / self.T_n

            return x_dot, n_dot

        return x_dot

    def u_step(self, u, prev_u):
        n = ca.vertcat(prev_u)

        return u
