import numpy as np
from utils import Smtrx, Hmtrx, D2R
import casadi as ca

# TODO: Add region constraints


class Model():
    """
    Base class for vehicle models
    """

    def __init__(self, dt, N) -> None:
        self.dt = dt
        self.N = N

    def _init_model(self) -> None:
        pass

    def step(self, x, u) -> None:
        pass

    def single_shooting(self, x_init: np.ndarray,
                        u_init: np.ndarray, opti: ca.Opti) -> tuple:
        pass

    def direct_collocation(self, x_init: np.ndarray,
                           u_init: np.ndarray, opti: ca.Opti) -> tuple:
        """
        Direct collocation method 

        Based on the work of Joel Andersson, Joris Gillis and Moriz Diehl at KU Leuven

        Links:
        https://github.com/casadi/casadi/blob/main/docs/examples/matlab/direct_collocation_opti.m
        and
        https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py

        Parameters
        ----------
            x_init : np.ndarray
                Initial 

        """
        pass

    def RK4(self, x: ca.Opti.variable, u: ca.Opti.variable,
            N: int, opti: ca.Opti) -> None:
        """
        Runga-Kutta 4 method for making discrete model constraints

        Parameters
        ----------
            x : ca.Opti.variable
                State variables to optimize
            u : ca.Opti.variable
                Control input variables to optimize
            N : int
                Number of RK steps
            opti : ca.Opti
                Optimizer object


        Returns
        -------
            None

        """

        for k in range(N):
            k1 = self.step(x[:, k],                  u[:, k])
            k2 = self.step(x[:, k] + self.dt/2 * k1, u[:, k])
            k3 = self.step(x[:, k] + self.dt/2 * k2, u[:, k])
            k4 = self.step(x[:, k] + self.dt * k3,   u[:, k])
            x_next = x[:, k] + self.dt/6 * (k1+2*k2+2*k3+k4)
            opti.subject_to(x[:, k+1] == x_next)


class DubinsCarModel(Model):
    def __init__(self, dt: float = 0.05, N: int = 40) -> None:
        super().__init__(dt, N)

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

        return ca.vertcat(u[0]*ca.cos(x[2]),
                          u[0]*ca.sin(x[2]),
                          u[0]*u[1])

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
        # for k in range(self.N):
        #     k1 = self.step(x[:, k],                  u[:, k])
        #     k2 = self.step(x[:, k] + self.dt/2 * k1, u[:, k])
        #     k3 = self.step(x[:, k] + self.dt/2 * k2, u[:, k])
        #     k4 = self.step(x[:, k] + self.dt * k3,   u[:, k])
        #     x_next = x[:, k] + self.dt/6 * (k1+2*k2+2*k3+k4)
        #     opti.subject_to(x[:, k+1] == x_next)
        self.RK4(x, u, self.N, opti)

        # TODO: Put this code in a separate method
        # ----------------------------------------------------
        # Control signal and time constraint
        opti.subject_to(opti.bounded(-1, v, 1))
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

    def direct_collocation(self, x_init: np.ndarray, u_init: np.ndarray, opti: ca.Opti) -> tuple:
        # Degree of interpolating polynomial
        d = 3

        # Get collocation points
        tau_root = np.append(0, ca.collocation_points(d, "legendre"))

        # Collocation, continuity abd quadrature coefficients
        C, D, B = np.zeros((d+1, d+1)), np.zeros(d+1), np.zeros(d+1)

        for j in range(d+1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(d+1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / \
                        (tau_root[j]-tau_root[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for r in range(d+1):
                C[j, r] = pder(tau_root[r])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            B[j] = pint(1.0)

        # Setup states
        x_pos = ca.SX.sym("x")
        y_pos = ca.SX.sym("y")
        theta = ca.SX.sym("theta")
        x = ca.vertcat(x_pos,
                       y_pos,
                       theta)

        # Setup inputs
        v = ca.SX.sym("v")
        phi = ca.SX.sym("phi")
        u = ca.vertcat(v,
                       phi)

        # Model
        xdot = self.step(x, u)


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
