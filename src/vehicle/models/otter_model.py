import numpy as np
import casadi as ca
import utils
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
        opti.subject_to(opti.bounded(self.n_min, u[0, :], self.n_max))
        opti.subject_to(opti.bounded(self.n_min, u[1, :], self.n_max))

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
        self.k_neg = 0.01289 / 2  # Negative Bollard, one propeller

        # max. prop. rev.
        self.n_max = np.sqrt((0.5 * 24.4 * self.g) / self.k_pos)
        # min. prop. rev.
        self.n_min = -np.sqrt((0.5 * 13.6 * self.g) / self.k_neg)

        # MRB_CG = [ (m+mp) * I2  O2      (Fossen 2021, Chapter 3)
        #               O2        Ig ]
        MRB_CG = np.zeros((3, 3))
        MRB_CG[0:2, 0:2] = (m + self.mp) * np.eye(2)

        # For 3-DOF, Ig = Iz
        MRB_CG[2:3, 2:3] = self.Ig[-1, -1]
        MRB = self.H_rg.T @ MRB_CG @ self.H_rg

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
        B = self.k_pos * np.array([[1, 1], [-self.l1, -self.l2]])
        self.Binv = np.linalg.inv(B)

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

        # Update thruster coefficients
        self.k_pos = model_params["k_pos"]
        self.k_neg = model_params["k_neg"]

    def step(self, x: ca.Opti.variable, prev_u: ca.Opti.variable,
             u: ca.Opti.variable) -> tuple[ca.Opti.variable, ca.Opti.variable]:
        """
        Step method
        [nu,u_feedback] = step(eta,nu,u_feedback,action,beta_c,V_c) integrates
        the Otter USV equations of motion using Euler's method.

        Parameters
        -----------
            x : ca.Opti.variable
                State space containing pose and velocity in 3-DOF
            prev_u : np.ndarray
                Previous control input
            action : np.ndarray
                Current control input

        Returns
        -------
            nu : ca.Opti.variable
                Updated velocity
            u : ca.Opti.variable
                Updated control input

        """

        # Split states into eta and nu
        eta = x[:3]
        nu = x[3:]

        # Input vector
        n = ca.vertcat(prev_u[0],
                       prev_u[1])

        # Rigid body and added mass Coriolis and centripetal matrices
        # CRB_CG = [ (m+mp) * Smtrx(nu2)          O3           (Fossen 2021, Chapter 6)
        #              O3                   -Smtrx(Ig*nu2)  ]
        # CRB_CG = np.zeros((3, 3))
        # CRB_CG[0:2, 0:2] = self.m_total * s
        # CRB_CG[-1, -1] = -Smtrx(np.matmul(self.Ig, nu[-1]))[-1, -1]

        # Make CRB directly for simplicty
        # -------------------------------------
        # CRB = self.H_rg.T @ CRB_CG @ self.H_rg  # transform CRB from CG to CO
        # # -------------------------------------

        # CA = utils.m2c(self.MA, nu)
        # print(f"CA with Munk moment: {CA}")
        # CA[-1, 0] = 0  # assume that the Munk moment in yaw can be neglected
        # CA[-1, 1] = 0  # if nonzero, must be balanced by adding nonlinear damping
        # CA[0, -1] = 0
        # CA[1, -1] = 0
        # print(f"CA without Munk moment: {CA}")

        C = utils.opt.m2c(self.M, nu)  # CRB + CA

        # Control forces and moments - with propeller revolution saturation
        # thrust = np.zeros(2)
        # for i in range(0, 2):

        # saturation, physical limits
        # TODO: Check if saturation constrain in _init_opt is
        #       sufficient or if we need something similar to
        #       the line below
        # ----------------------------------------------
        # n[i] = utils.sat(self.n_min, n[i], self.n_max)
        # ----------------------------------------------

        # if n[i] > 0:  # positive thrust
        #     thrust[i] = self.k_pos * n[i] * abs(n[i])
        # else:  # negative thrust
        #     thrust[i] = self.k_neg * n[i] * abs(n[i])
        thrust = ca.vertcat(self.k_pos * n[0] * ca.fabs(n[0]),
                            self.k_pos * n[1] * ca.fabs(n[1]))

        # Control forces and moments
        tau = ca.vertcat(thrust[0] + thrust[1],
                         0,
                         -self.l1 * thrust[0] - self.l2 * thrust[1])
        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = -self.D @ nu
        tau_damp[-1] = tau_damp[-1] - 10 * \
            self.D[-1, -1] * ca.fabs(nu[-1]) * nu[-1]

        # State derivatives (with dimension)
        tau_crossflow = utils.opt.crossFlowDrag(
            self.L, self.B_pont, self.T, nu)

        sum_tau = (
            tau
            + tau_damp
            + tau_crossflow
            - C @ nu
        )

        nu_dot = self.Minv @ sum_tau
        eta_dot = utils.opt.Rz(eta[-1]) @ nu
        u_dot = (u - n) / self.T_n  # propeller dynamics

        x_dot = ca.vertcat(eta_dot,
                           nu_dot)

        return x_dot, u_dot

    def _denormalise(self, u):
        if u[0] < 0:
            u[0] = u[0]*111
        else:
            u[0] = u[0]*113

        if u[1] < 0:
            u[1] = u[1]*111
        else:
            u[1] = u[1]*113

        return u

    # def single_shooting(self, x_init, u_init, opti: ca.Opti):
    #     # Declaring optimization variables
    #     # State variables
    #     x = opti.variable(3, self.N+1)
    #     x_pos = x[0, :]
    #     y_pos = x[1, :]
    #     theta = x[2, :]

    #     # Input variables
    #     u = opti.variable(2, self.N)
    #     v = u[0, :]
    #     phi = u[1, :]

    #     # Slack variables
    #     s = opti.variable(3, self.N+1)

    #     # Fixed step Runge-Kutta 4 integrator
    #     self.RK4(x, u, self.N, opti)

    #     # TODO: Put this code in a separate method
    #     # ----------------------------------------------------
    #     # Control signal and time constraint
    #     opti.subject_to(opti.bounded(-1, v, 2))
    #     opti.subject_to(opti.bounded(D2R(-15), phi, D2R(15)))
    #     # ----------------------------------------------------

    #     # Boundary values
    #     # Initial conditions
    #     opti.subject_to(x_pos[0] == x_init[0])
    #     opti.subject_to(y_pos[0] == x_init[1])
    #     opti.subject_to(theta[0] == x_init[2])
    #     opti.subject_to(v[0] == u_init[0])
    #     opti.subject_to(phi[0] == u_init[1])

    #     # Initial guesses
    #     opti.set_initial(x_pos, x_init[0])
    #     opti.set_initial(y_pos, x_init[1])
    #     opti.set_initial(theta, x_init[2])
    #     opti.set_initial(v, u_init[0])
    #     opti.set_initial(phi, u_init[1])

    #     return x, u, s
