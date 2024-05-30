import numpy as np
import casadi as ca
import utils
import utils.opt
from .models import Model

# TODO: Rearrange slack vector


class OtterModel(Model):
    def __init__(self, dt: float = 0.05, N: int = 40, buffer: float = 0.2, default=False,
                 estimate_current: bool = False) -> None:
        super().__init__(dt, N)
        self._init_model(default)

        # Make vessel safety boundary
        half_length = self.L/2 + buffer
        half_beam = self.B/2 + buffer
        self.safety_bounds = np.array([[half_length, half_beam],
                                       [half_length, -half_beam],
                                       [-half_length, -half_beam],
                                       [-half_length, half_beam]])

        self.estimate_current = estimate_current

        # Impose speedlimit
        self.speed_limit = utils.kts2ms(5)

    def _init_opt(self, x_init, u_init, opti: ca.Opti, space: np.ndarray = None):
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
        slack = opti.variable(7, self.N+1)

        # Spatial constraints
        if space is not None:
            A, b = space
            for k in range(1, self.N+1):
                for bound in self.safety_bounds:
                    # State pos constraint
                    opti.subject_to(
                        A @ (utils.opt.R(x[2, k]) @ bound +
                             x[:2, k] - slack[:2, k]) <= b
                    )

        # Control signal and time constraint
        opti.subject_to(opti.bounded(-70 - slack[2, :-1],
                                     port_u,
                                     100 + slack[2, :-1]))
        opti.subject_to(opti.bounded(-70 - slack[3, :-1],
                                     starboard_u,
                                     100 + slack[3, :-1]))
        opti.subject_to(opti.bounded(-self.dt*100 - slack[4, 1:-1],
                                     port_u[:, 1:] - port_u[:, :-1],
                                     self.dt*100 + slack[4, 1:-1]))
        opti.subject_to(opti.bounded(-self.dt*100 - slack[5, 1:-1],
                                     starboard_u[:, 1:] - starboard_u[:, :-1],
                                     self.dt*100 + slack[5, 1:-1]))

        opti.subject_to(opti.bounded(0, slack, np.inf))

        # for k in range(self.N):
        #     opti.subject_to(opti.bounded(utils.kts2ms(-3) - slack[6, k],
        #                                  surge[k],
        #                                  utils.kts2ms(3)) + slack[6, k])
        # opti.subject_to(opti.bounded(utils.kts2ms(-3),
        #                              surge,
        #                              utils.kts2ms(3)))

        # Boundary values
        # Initial conditions
        opti.subject_to(north[0] == x_init[0])
        opti.subject_to(east[0] == x_init[1])
        opti.subject_to(yaw[0] == x_init[2])
        opti.subject_to(surge[0] == x_init[3])
        opti.subject_to(sway[0] == x_init[4])
        opti.subject_to(yaw_rate[0] == x_init[5])
        opti.subject_to(port_u[0] == u_init[0])
        opti.subject_to(starboard_u[0] == u_init[1])

        opti.set_initial(north, x_init[0])
        opti.set_initial(east, x_init[1])
        opti.set_initial(yaw, x_init[2])
        opti.set_initial(port_u, u_init[0])
        opti.set_initial(starboard_u, u_init[1])

        return x, u, slack

    def _init_model(self, default):
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
        self.mp = 0  # 25.0                           # Payload (kg)
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
        Xu = -24.4 * self.g / Umax  # specified using the maximum speed
        # specified using the time constant in sway
        Yv = -self.M[1, 1] / T_sway
        # Nr will be different from the 6-DOF model
        Nr = -self.M[-1, -1] / T_yaw  # specified by the time constant T_yaw

        # Linear damping
        self.D = -np.diag([Xu, Yv, Nr])

        # Nonlinear damping
        self.Nrr = 10 * Nr  # negative

        # Propeller configuration/input matrix
        # TODO: This is probably wrong
        B = np.array([[self.k_port, 0],
                      [0, self.k_stb]]).dot(np.array([[1, 1], [-self.l1, -self.l2]]))
        self.Binv = np.linalg.inv(B)

        # Cross-sectional weights see Martinsen
        self.W = ca.diag([self.B, self.L, 1])

        # Environment forces in NED
        self.w = np.zeros(3)

        # Learning cost parameters
        self.l = 0  # Initial cost parameters
        self.v = np.zeros(3, float)  # Terminal cost parameters

        self.m_off_diag = self.m_total*self.xg

        self.original_theta = np.array(
            [
                self.m_total, self.Ig[-1, -1], self.m_off_diag,
                Xudot, Yvdot, Nrdot, Xu, Yv, Nr, self.Nrr,
                self.k_port, self.k_stb,
                0, 0, 0,    # Environment vector
                0,          # Initial cost
                0, 0, 0     # Terminal cost
            ]
        )

        if default:
            self.theta = self.original_theta.copy()
        else:
            # self.theta = np.array(
            #     [
            #         0.9*self.m_total, 0.9*self.Ig[-1, -1], 0.9*self.xg,
            #         0.9*Xudot, 0.9*Yvdot, 0.9*Nrdot, 0.9*Xu, 0.9*Yv, 0.9*Nr, 0.9*self.Nrr,
            #         0.9*self.k_port, 0.9*self.k_stb,
            #         0, 0, 0,    # Environment vector
            #         1,          # Initial cost
            #         1, 1, 1     # Terminal cost
            #     ]
            # )
            self.theta = np.array(
                [
                    0.9*self.m_total, 0.9 *
                    self.Ig[-1, -1], 0.9*self.m_off_diag,
                    0.9*Xudot, 0.9*Yvdot, 0.9*Nrdot, 0.9*Xu, 0.9*Yv, 0.9*Nr, 0.9*self.Nrr,
                    self.k_port, self.k_stb,
                    0, 0, 0,    # Environment vector
                    1,          # Initial cost
                    1, 1, 1     # Terminal cost
                ]
            )

        print(f"initial theta in model: {np.round(self.theta, 5)}")

    def _update(self, theta) -> None:
        """
        Update model parameters
        Takes in model parameters from another place and updates

        Parameters
        -----------
            theta : Any
                Model and cost parameters, 
                can be ca.Opti.parameter or np.ndarray

        Returns
        -------
            self
        """

        # Rigid body mass
        self.m_total = theta[0]  # model_params["m_total"]
        Iz = theta[1]
        # TODO: determine if this should be estimated
        # self.xg = theta[2]
        self.m_off_diag = theta[2]

        # MRB = np.zeros((3, 3))
        # MRB = np.array([[self.m_total, 0, self.m_total * self.xg],
        #                 [0, self.m_total, self.m_total * self.xg],
        #                 [0, self.m_total * self.xg, Iz]])
        MRB = ca.MX.zeros(3, 3)
        MRB[0, 0] = self.m_total
        MRB[1, 1] = self.m_total
        MRB[1, 2] = self.m_off_diag  # self.m_total * self.xg
        MRB[2, 1] = MRB[1, 2]
        MRB[2, 2] = Iz

        # Update hydrodynamic mass
        added_mass = theta[3:6]
        self.MA = -ca.diag(added_mass)

        # Update mass and damping coefficients
        self.M = MRB + self.MA
        self.Minv = ca.inv(self.M)

        # Update linear damping
        self.D = -ca.diag(theta[6:9])

        # Nonlinear damping in yaw
        self.Nrr = theta[9]

        # Update thruster coefficients
        self.k_port = theta[10]
        self.k_stb = theta[11]
        # self.k_port = self.k_pos
        # self.k_stb = self.k_pos
        # B = np.array([[self.k_port, self.k_stb],
        #               [0, 0],
        #               [-self.k_port*self.l1, -self.k_stb*self.l1]])

        # Environment vector
        self.w = theta[12:15]
        self.w[-1] = 0  # TODO: Determine if this is correct

        # Learning cost parameters
        self.l = theta[15]      # Initial cost parameters
        self.v = theta[16:]     # Terminal cost parameters

        if False:
            print("======== Update ========")
            print(f"self.M:     {self.M}")
            print(f"self.D:     {self.D}")
            print(f"self.Nrr:   {self.Nrr}")
            # print(f"B:          {B}")
            print(f"self.w:     {self.w}")
            print(f"self.l:     {self.l}")
            print(f"self.v:     {self.v}")

    def step(self, x: ca.Opti.variable, u: ca.Opti.variable, prev_u: ca.Opti.variable = None, rl=False) -> tuple[ca.Opti.variable, ca.Opti.variable]:
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
        # Split states into eta and nu
        eta = x[:3]
        nu = x[3:]

        # Input vector
        n = [u[0], u[1]]

        # ===============
        # Coriolis matrix
        # ===============
        # CRB based on assumptions from
        # Fossen 2021, Chapter 6, page 137
        CRB = ca.MX.zeros(3, 3)
        CRB[0, 1] = -self.m_total * nu[2]
        CRB[0, 2] = -self.m_off_diag * nu[2]  # self.m_total * self.xg * nu[2]
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
                            self.k_stb * n[1]*ca.fabs(n[1]))

        # Control forces and moments
        tau = ca.vertcat(thrust[0] + thrust[1],
                         0,
                         -self.l1 * thrust[0] - self.l2 * thrust[1])

        # ================
        # Calculate forces
        # ================
        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = self.D @ nu  # pos
        tau_damp[2] = tau_damp[2] - self.Nrr * ca.fabs(nu[2]) * nu[2]  # pos

        # =========================
        # Solve the Fossen equation
        # =========================
        sum_tau = (
            tau
            - tau_damp  # neg
            - C @ nu
        )

        # Add environmental forces when using RL
        if rl and self.estimate_current:
            sum_tau += self.W @ utils.opt.Rz(eta[2]).T @ self.w

        # ==================
        # Calculate dynamics
        # ==================
        # Transform nu from {b} to {n}
        eta_dot = utils.opt.Rz(eta[2]) @ nu
        nu_dot = self.Minv @ sum_tau

        # Construct state vector
        x_dot = ca.vertcat(eta_dot,
                           nu_dot)

        return x_dot

    def implicit(self, x: ca.Opti.variable, u: ca.Opti.variable, x_next: ca.Opti.variable, dt=None, rl=False) -> ca.MX:
        """
        Implicit model method
        Defines vessel model and discretizes it using forward Euler

        Parameters
        -----------
            x : ca.Opti.variable
                State space containing pose and velocity in 3-DOF
            u : np.ndarray
                Current control input
            x_next : ca.Opti.variable
                Next time step

        Returns
        -------
            implicit_model : ca.Opti.variable
                Model equations

        """

        if dt is not None:
            self.dt = dt

        # =======================
        # Prep decision variables
        # =======================
        # Split states into eta and nu
        eta = x[:3]
        nu = x[3:]

        eta_dot = x_next[:3]
        nu_dot = x_next[3:]

        # Input vector
        n = [u[0], u[1]]

        # ===============
        # Coriolis matrix
        # ===============
        # CRB based on assumptions from
        # Fossen 2021, Chapter 6, page 137
        CRB = ca.MX.zeros(3, 3)
        CRB[0, 1] = -self.m_total * nu[2]
        CRB[0, 2] = -self.m_off_diag * nu[2]  # self.m_total * self.xg * nu[2]
        CRB[1, 0] = -CRB[0, 1]
        CRB[2, 0] = -CRB[0, 2]

        # Added coriolis with Munk moment
        CA = utils.opt.m2c(self.MA, nu)

        C = CRB + CA

        # ======================
        # Thrust dynamics
        # ======================
        # thrust = n
        thrust = ca.vertcat(self.k_port * n[0]*ca.sqrt(n[0]**2),
                            self.k_stb * n[1]*ca.sqrt(n[1]**2))

        # Control forces and moments
        tau = ca.vertcat(thrust[0] + thrust[1],
                         0,
                         -self.l1 * thrust[0] - self.l2 * thrust[1])

        # ================
        # Calculate forces
        # ================
        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = self.D @ nu  # pos
        tau_damp[2] = tau_damp[2] - self.Nrr * ca.sqrt(nu[2]**2) * nu[2]  # pos

        # ==================
        # Calculate dynamics
        # ==================
        kinematics = eta_dot - self.dt*utils.opt.Rz(eta[2]) @ nu

        # Fossen equation
        kinetics = (
            self.M @ nu_dot
            + self.dt * tau_damp  # pos pos
            + self.dt*C @ nu
            - self.dt*tau
        )

        # Add environmental forces when using RL
        if rl and self.estimate_current:
            kinetics -= self.dt*self.W @ utils.opt.Rz(eta[2]).T @ self.w

        # Construct model vector
        implicit_model = ca.vertcat(kinematics,
                                    kinetics)

        return implicit_model

    def forward_step(self, x_init, u_init, dt) -> np.ndarray:
        """
        Step method
        [nu,u_feedback] = step(eta,nu,u_feedback,action,beta_c,V_c) integrates
        the Otter USV equations of motion using Euler's method.

        Parameters
        -----------
            x : np.ndarray
                State space containing pose and velocity in 3-DOF
            u : np.ndarray
                Current control input

        Returns
        -------
            x_next : np.ndarray
                Derivative of eta and nu


        """

        x_next = utils.RK4(x_init, u_init, dt, self._ode)

        return x_next

    def _ode(self, x, u) -> np.ndarray:
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
        # Split states into eta and nu
        eta = x[:3]
        nu = x[3:]

        # Input vector
        n = [u[0], u[1]]

        # ===============
        # Coriolis matrix
        # ===============
        # CRB based on assumptions from
        # Fossen 2021, Chapter 6, page 137
        CRB = np.zeros((3, 3))
        CRB[0, 1] = -self.m_total * nu[2]
        CRB[0, 2] = -self.m_total * self.xg * nu[2]
        CRB[1, 0] = -CRB[0, 1]
        CRB[2, 0] = -CRB[0, 2]

        # Added coriolis with Munk moment
        CA = utils.m2c(self.MA, nu)
        C = CRB + CA

        # ======================
        # Thrust dynamics
        # ======================
        thrust = np.array(
            [
                self.k_port * n[0]*abs(n[0]),
                self.k_stb * n[1]*abs(n[1])
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
        tau_damp = -self.D @ nu
        tau_damp[2] = tau_damp[2] - self.Nrr * abs(nu[2]) * nu[2]

        # =========================
        # Solve the Fossen equation
        # =========================
        sum_tau = (
            tau
            - tau_damp
            - C @ nu
        )

        # ==================
        # Calculate dynamics
        # ==================
        # Transform nu from {b} to {n}
        eta_dot = utils.Rz(eta[2]) @ nu
        nu_dot = self.Minv @ sum_tau
        nu_dot = nu_dot.reshape(3,)

        # Construct state vector
        x_dot = np.concatenate([eta_dot, nu_dot])

        return x_dot

    def direct_collocation(self, x_init, u_init, x_d, config, opti: ca.Opti, space: np.ndarray = None):
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
                Initial state

        """

        # Degree of interpolating polynomial
        d = 3

        # Get collocation points
        tau_root = np.append(0, ca.collocation_points(d, 'legendre'))

        # Coefficients of the collocation equation
        C = np.zeros((d+1, d+1))

        # Coefficients of the continuity equation
        D = np.zeros(d+1)

        # Coefficients of the quadrature function
        B = np.zeros(d+1)

        # Construct polynomial basis
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

        # Declaring optimization variables
        x, u, slack = self._init_opt(x_init, u_init, opti, space)

        # Start with an initial cost
        J = 0

        # Formulate the NLP
        for k in range(self.N):
            # ===========================
            # State at collocation points
            # ===========================
            Xc = opti.variable(6, d)

            opti.subject_to(opti.bounded(utils.kts2ms(-5) - slack[6, k],
                                         Xc[3, :],
                                         utils.kts2ms(5) + slack[6, k]))
            opti.subject_to(opti.bounded(-np.pi, Xc[5, :], np.pi))

            opti.subject_to(opti.bounded(-np.inf, Xc, np.inf))

            # ============================
            # Loop over collocation points
            # ============================
            Xk_end = D[0]*x[:, k]
            for j in range(1, d+1):
                opti.set_initial(Xc[:, j-1], x_init)

                # Expression for the state derivative at the collocation point
                xp = C[0, j]*x[:, k]
                for r in range(d):
                    xp = xp + C[r+1, j]*Xc[:, r]

                # Collocation state dynamics
                implicit = self.implicit(Xc[:, j-1], u[:, k], xp)

                # Collocation objective function contribution
                qj = utils.opt.pseudo_huber(
                    Xc[:, j-1],
                    u[:, k],
                    x_d,
                    config,
                    slack[:, k]
                )

                # Apply dynamics with forward euler
                # this is where the dynamics integration happens
                # opti.subject_to(self.dt*fj == xp)
                opti.subject_to(implicit == 0)

                # Add contribution to the end state
                Xk_end = Xk_end + D[j]*Xc[:, j-1]

                # Add contribution to quadrature function using forward euler
                # this is where the objective integration happens
                J = J + B[j]*qj*self.dt

            # Add equality constraint
            opti.subject_to(x[:, k+1] == Xk_end)

        # Minimize objective
        opti.minimize(J)

        return x, u, slack, J

    def rl_step(self, x_init, u_init, x_d, new_theta,
                config, opti: ca.Opti, space: np.ndarray = None, step_type="Q"):
        """
        RL step method

        Based on the work of Joel Andersson, Joris Gillis and Moriz Diehl at KU Leuven

        Links:
        https://github.com/casadi/casadi/blob/main/docs/examples/matlab/direct_collocation_opti.m
        and
        https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py

        Parameters
        ----------
            x_init : np.ndarray
                Initial state

        """

        gamma = config["gamma"]
        speed_limit = config["speed limit"]

        # Degree of interpolating polynomial
        d = 3

        # Get collocation points
        tau_root = np.append(0, ca.collocation_points(d, 'legendre'))

        # Coefficients of the collocation equation
        C = np.zeros((d+1, d+1))

        # Coefficients of the continuity equation
        D = np.zeros(d+1)

        # Coefficients of the quadrature function
        B = np.zeros(d+1)

        # Construct polynomial basis
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

        # Declaring optimization variables
        x, u, slack = self._init_opt(x_init, u_init, opti, space)

        # Theta must be a opti parameter
        theta = opti.parameter(16+3)
        opti.set_value(theta, new_theta)
        self._update(theta=theta)

        # Remove condition on u0 in value-function step
        if step_type == "V":
            opti.subject_to(opti.bounded(-np.inf, u[:, 0], np.inf))

        # Start with an empty objective function
        initial_cost = theta[15]
        J = initial_cost

        dual_list = []
        model_constraint_list = []

        # Formulate the NLP
        for k in range(self.N):
            # ===========================
            # State at collocation points
            # ===========================
            Xc = opti.variable(6, d)

            opti.subject_to(opti.bounded(utils.kts2ms(-speed_limit) - slack[6, k],
                                         Xc[3, :],
                                         utils.kts2ms(speed_limit) + slack[6, k]))
            opti.subject_to(opti.bounded(-np.pi, Xc[5, :], np.pi))

            opti.subject_to(opti.bounded(-np.inf, Xc, np.inf))

            # ============================
            # Loop over collocation points
            # ============================
            Xk_end = D[0]*x[:, k]
            for j in range(1, d+1):
                opti.set_initial(Xc[:, j-1], x_init)

                # Expression for the state derivative at the collocation point
                xp = C[0, j]*x[:, k]
                for r in range(d):
                    xp = xp + C[r+1, j]*Xc[:, r]

                # Collocation state dynamics
                implicit = self.implicit(Xc[:, j-1], u[:, k], xp, rl=True)

                # Collocation objective function contribution
                if x_d.ndim == 2:
                    qj = utils.opt.pseudo_huber(
                        Xc[:, j-1],
                        u[:, k],
                        x_d[:, k],
                        config,
                        slack[:, k]
                    )
                else:
                    qj = utils.opt.pseudo_huber(
                        Xc[:, j-1],
                        u[:, k],
                        x_d,
                        config,
                        slack[:, k]
                    )

                # Multiply by gamma^i
                qj = pow(gamma, k)*qj

                # Apply dynamics with forward euler
                # this is where the dynamics integration happens
                model_constraint = implicit == 0
                opti.subject_to(model_constraint)
                dual_list.append(opti.dual(model_constraint))
                model_constraint_list.append(implicit)

                # Add contribution to the end state
                Xk_end = Xk_end + D[j]*Xc[:, j-1]

                # Add contribution to quadrature function using forward euler
                # this is where the objective integration happens
                J = J + B[j]*qj*self.dt

            # Add equality constraint
            opti.subject_to(x[:, k+1] == Xk_end)

        if not ca.MX.is_zero(self.v):
            # Find and add terminal cost
            if x_d.ndim == 2:
                terminal_error = x[:3, -1] - x_d[:3, -1]
            else:
                terminal_error = x[:3, -1] - x_d

            terminal_cost = pow(gamma, self.N) * (
                terminal_error.T @ ca.diag(theta[16:]) @ terminal_error
            )

            J += terminal_cost

        # Minimize objective
        opti.minimize(J)

        # Calculate needed gradiens if estimating Q-function
        if step_type == "Q":
            # Make Lagrangian function
            dual = ca.vertcat(*dual_list)
            model_constraint = ca.vertcat(*model_constraint_list)
            Lagrangian = initial_cost + terminal_cost - dual.T @ model_constraint
            # Lagrangian = - dual.T @ model_constraint

            # Calculate gradient of Q
            grad = ca.gradient(Lagrangian, theta)

            # Calculate gradient of f
            # grad_f = ca.gradient(model_constraint[0], theta)
            grad_f = ca.jacobian(self.step(x[:, 0], u[:, 0], rl=True), theta)

            return x, u, slack, theta, J, grad, grad_f, Lagrangian

        # Return fewer quantities if estimating value function
        elif step_type == "V":
            return u, J, x, u, slack

        else:
            raise Exception(f"{step_type} is not a valid step type")

    # def V_step(self, x_init, u_init, x_d, new_theta, config, opti: ca.Opti, space: np.ndarray = None):
    #     """
    #     Value-function step method

    #     Based on the work of Joel Andersson, Joris Gillis and Moriz Diehl at KU Leuven

    #     Links:
    #     https://github.com/casadi/casadi/blob/main/docs/examples/matlab/direct_collocation_opti.m
    #     and
    #     https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py

    #     Parameters
    #     ----------
    #         x_init : np.ndarray
    #             Initial state

    #     """

    #     # Degree of interpolating polynomial
    #     d = 3

    #     # Get collocation points
    #     tau_root = np.append(0, ca.collocation_points(d, 'legendre'))

    #     # Coefficients of the collocation equation
    #     C = np.zeros((d+1, d+1))

    #     # Coefficients of the continuity equation
    #     D = np.zeros(d+1)

    #     # Coefficients of the quadrature function
    #     B = np.zeros(d+1)

    #     # Construct polynomial basis
    #     for j in range(d+1):
    #         # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    #         p = np.poly1d([1])
    #         for r in range(d+1):
    #             if r != j:
    #                 p *= np.poly1d([1, -tau_root[r]]) / \
    #                     (tau_root[j]-tau_root[r])

    #         # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    #         D[j] = p(1.0)

    #         # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    #         pder = np.polyder(p)
    #         for r in range(d+1):
    #             C[j, r] = pder(tau_root[r])

    #         # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
    #         pint = np.polyint(p)
    #         B[j] = pint(1.0)

    #     # Declaring optimization variables
    #     x, u, slack = self._init_opt(x_init, u_init, opti, space)

    #     # Remove condition on u0
    #     opti.subject_to(opti.bounded(-np.inf, u[:, 0], np.inf))

    #     # Theta
    #     theta = opti.parameter(16+3)
    #     opti.set_value(theta, new_theta)
    #     self._update(theta=theta)

    #     initial_cost = theta[15]
    #     J = initial_cost

    #     # Formulate the NLP
    #     for k in range(self.N):
    #         # ===========================
    #         # State at collocation points
    #         # ===========================
    #         Xc = opti.variable(6, d)

    #         # # Spatial constraints
    #         # if space is not None:
    #         #     A, b = space
    #         #     for j in range(d):
    #         #         # State pos constraint
    #         #         opti.subject_to(A @ Xc[:2, j] <= b)

    #         opti.subject_to(opti.bounded(utils.kts2ms(-5) - slack[6, k],
    #                                      Xc[3, :],
    #                                      utils.kts2ms(5) + slack[6, k]))
    #         opti.subject_to(opti.bounded(-np.pi, Xc[5, :], np.pi))

    #         opti.subject_to(opti.bounded(-np.inf, Xc, np.inf))

    #         # ============================
    #         # Loop over collocation points
    #         # ============================
    #         Xk_end = D[0]*x[:, k]
    #         for j in range(1, d+1):
    #             opti.set_initial(Xc[:, j-1], x_init)

    #             # Expression for the state derivative at the collocation point
    #             xp = C[0, j]*x[:, k]
    #             for r in range(d):
    #                 xp = xp + C[r+1, j]*Xc[:, r]

    #             # Collocation state dynamics
    #             implicit = self.implicit(Xc[:, j-1], u[:, k], xp, rl=True)

    #             # Collocation objective function contribution
    #             if x_d.ndim == 2:
    #                 qj = utils.opt.pseudo_huber(
    #                     Xc[:, j-1],
    #                     u[:, k],
    #                     x_d[:, k],
    #                     config,
    #                     slack[:, k]
    #                 )
    #             else:
    #                 qj = utils.opt.pseudo_huber(
    #                     Xc[:, j-1],
    #                     u[:, k],
    #                     x_d,
    #                     config,
    #                     slack[:, k]
    #                 )

    #             # Multiply by gamma^i
    #             qj = pow(gamma, k)*qj

    #             # Apply dynamics with forward euler
    #             # this is where the dynamics integration happens
    #             opti.subject_to(implicit == 0)

    #             # Add contribution to the end state
    #             Xk_end = Xk_end + D[j]*Xc[:, j-1]

    #             # Add contribution to quadrature function using forward euler
    #             # this is where the objective integration happens
    #             J = J + B[j]*qj*self.dt

    #         # Add equality constraint
    #         opti.subject_to(x[:, k+1] == Xk_end)

    #     if not ca.MX.is_zero(self.v):
    #         # Find and add terminal cost
    #         if x_d.ndim == 2:
    #             terminal_error = x[:3, -1] - x_d[:3, -1]
    #         else:
    #             terminal_error = x[:3, -1] - x_d
    #         terminal_cost = config["gamma"] * (
    #             terminal_error.T @ ca.diag(theta[16:]) @ terminal_error
    #         )

    #         J += terminal_cost

    #     # Minimize objective
    #     opti.minimize(J)

    #     return u, J, x, u, slack

    # def as_direct_collocation(self, x_init, u_init, x_d, config, opti: ca.Opti, space: np.ndarray = None):
    #     """
    #     Advanced step direct collocation method

    #     Based on the work of Joel Andersson, Joris Gillis and Moriz Diehl at KU Leuven

    #     Links:
    #     https://github.com/casadi/casadi/blob/main/docs/examples/matlab/direct_collocation_opti.m
    #     and
    #     https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py

    #     Parameters
    #     ----------
    #         x_init : np.ndarray
    #             Initial state

    #     """

    #     # TODO: Investigate if this is more stable than normal direct collocation

    #     # Degree of interpolating polynomial
    #     d = 3

    #     # Get collocation points
    #     tau_root = np.append(0, ca.collocation_points(d, 'legendre'))

    #     # Coefficients of the collocation equation
    #     C = np.zeros((d+1, d+1))

    #     # Coefficients of the continuity equation
    #     D = np.zeros(d+1)

    #     # Coefficients of the quadrature function
    #     B = np.zeros(d+1)

    #     # Construct polynomial basis
    #     for j in range(d+1):
    #         # Construct Lagrange polynomials to get the polynomial basis at the collocation point
    #         p = np.poly1d([1])
    #         for r in range(d+1):
    #             if r != j:
    #                 p *= np.poly1d([1, -tau_root[r]]) / \
    #                     (tau_root[j]-tau_root[r])

    #         # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    #         D[j] = p(1.0)

    #         # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    #         pder = np.polyder(p)
    #         for r in range(d+1):
    #             C[j, r] = pder(tau_root[r])

    #         # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
    #         pint = np.polyint(p)
    #         B[j] = pint(1.0)

    #     # Declaring optimization variables
    #     # State variables
    #     x = opti.variable(6, self.N+1)
    #     north = x[0, :]
    #     east = x[1, :]
    #     yaw = x[2, :]
    #     surge = x[3, :]
    #     sway = x[4, :]
    #     yaw_rate = x[5, :]

    #     # Input variables
    #     u = opti.variable(2, self.N)
    #     port_u = u[0, :]
    #     starboard_u = u[1, :]

    #     # Slack variables
    #     slack = opti.variable(7, self.N+1)

    #     safety_bounds = 1.1*np.array([[self.L/2, self.B/2],
    #                                   [self.L/2, -self.B/2],
    #                                   [-self.L/2, -self.B/2],
    #                                   [-self.L/2, self.B/2]])

    #     # Spatial constraints
    #     if space is not None:
    #         A, b = space
    #         for k in range(1, self.N+1):
    #             for bound in safety_bounds:
    #                 # State pos constraint
    #                 opti.subject_to(
    #                     A @ (utils.opt.R(x[2, k]) @ bound +
    #                          x[:2, k] - slack[:2, k]) <= b
    #                 )

    #     # Control signal and time constraint
    #     opti.subject_to(opti.bounded(-70 - slack[2, :-1],
    #                                  port_u,
    #                                  100 + slack[2, :-1]))
    #     opti.subject_to(opti.bounded(-70 - slack[3, :-1],
    #                                  starboard_u,
    #                                  100 + slack[3, :-1]))
    #     opti.subject_to(opti.bounded(-self.dt*100 - slack[4, 1:-1],
    #                                  port_u[:, 1:] - port_u[:, :-1],
    #                                  self.dt*100 + slack[4, 1:-1]))
    #     opti.subject_to(opti.bounded(-self.dt*100 - slack[5, 1:-1],
    #                                  starboard_u[:, 1:] - starboard_u[:, :-1],
    #                                  self.dt*100 + slack[5, 1:-1]))

    #     opti.subject_to(opti.bounded(0, slack, np.inf))

    #     # Calculate one state in the future using u0
    #     x_1 = utils.RK4(x_init, u_init, self.dt, self._ode)

    #     # Boundary values
    #     # Initial conditions
    #     opti.subject_to(north[0] == x_init[0])
    #     opti.subject_to(east[0] == x_init[1])
    #     opti.subject_to(yaw[0] == x_init[2])
    #     opti.subject_to(surge[0] == x_init[3])
    #     opti.subject_to(sway[0] == x_init[4])
    #     opti.subject_to(yaw_rate[0] == x_init[5])
    #     opti.subject_to(north[1] == x_1[0])
    #     opti.subject_to(east[1] == x_1[1])
    #     opti.subject_to(yaw[1] == x_1[2])
    #     opti.subject_to(surge[1] == x_1[3])
    #     opti.subject_to(sway[1] == x_1[4])
    #     opti.subject_to(yaw_rate[1] == x_1[5])

    #     opti.subject_to(port_u[0] == u_init[0])
    #     opti.subject_to(starboard_u[0] == u_init[1])

    #     # Initial guesses on solution
    #     opti.set_initial(north, x_1[0])
    #     opti.set_initial(east, x_1[1])
    #     opti.set_initial(yaw, x_1[2])
    #     opti.set_initial(surge, x_1[3])
    #     opti.set_initial(sway, x_1[4])
    #     opti.set_initial(yaw_rate, x_1[5])
    #     opti.set_initial(port_u, u_init[0])
    #     opti.set_initial(starboard_u, u_init[1])

    #     # Start with an initial cost
    #     J = 0

    #     # Formulate the NLP
    #     for k in range(1, self.N):
    #         # ===========================
    #         # State at collocation points
    #         # ===========================
    #         Xc = opti.variable(6, d)

    #         # TODO: Reformulate spatial constraints to
    #         #       include a safe boundary around the vessel
    #         # Spatial constraints
    #         if space is not None:
    #             A, b = space
    #             for j in range(d):
    #                 # State pos constraint
    #                 opti.subject_to(A @ Xc[:2, j] <= b)

    #         opti.subject_to(opti.bounded(utils.kts2ms(-5),
    #                                      Xc[3:5, :],
    #                                      utils.kts2ms(5)))
    #         opti.subject_to(opti.bounded(-np.pi, Xc[5, :], np.pi))

    #         # ============================
    #         # Loop over collocation points
    #         # ============================
    #         Xk_end = D[0]*x[:, k]
    #         for j in range(1, d+1):
    #             opti.set_initial(Xc[:, j-1], x_init)

    #             # Expression for the state derivative at the collocation point
    #             xp = C[0, j]*x[:, k]
    #             for r in range(d):
    #                 xp = xp + C[r+1, j]*Xc[:, r]

    #             # Collocation state dynamics
    #             # fj = self.step(Xc[:, j-1], u[:, k])
    #             implicit = self.implicit(Xc[:, j-1], u[:, k], xp)

    #             # Collocation objective function contribution
    #             qj = utils.opt.pseudo_huber(
    #                 Xc[:, j-1],
    #                 u[:, k],
    #                 x_d,
    #                 config,
    #                 slack[:, k]
    #             )

    #             # Apply dynamics with forward euler
    #             # this is where the dynamics integration happens
    #             opti.subject_to(implicit == 0)

    #             # Add contribution to the end state
    #             Xk_end = Xk_end + D[j]*Xc[:, j-1]

    #             # Add contribution to quadrature function using forward euler
    #             # this is where the objective integration happens
    #             J = J + B[j]*qj*self.dt

    #         # Add equality constraint
    #         opti.subject_to(x[:, k+1] == Xk_end)

    #     # Minimize objective
    #     opti.minimize(J)

    #     return x, u, slack
