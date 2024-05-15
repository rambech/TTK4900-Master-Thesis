import numpy as np

import utils.opt
from .control import Control
from .optimizer import Optimizer
from vehicle.models import Model, OtterModel
import casadi as ca
from plotting import plot_solution
import utils

# TODO: Determine the minimum controller speed needed,
#       in order to control the Otter


class NMPC(Control):
    """
    Nonlinear Model Predictive Control class
    """

    def __init__(self, model: Model, dof: int = 2,
                 config: dict = None, space: tuple = None,
                 use_slack: bool = False) -> None:
        """
        Parameters
        ----------
        model : Model
            Model for OCP constraints
        dof : int
            Degrees of freedom, default 2
        config : dict
            Optimization configuration, default None
        """

        super().__init__(dof)
        # If config not given use standard parameters
        if config != None:
            self.config = config
        else:
            self.config = {"N": 40,
                           "dt": 0.05,
                           "Q": np.diag([1, 1, 1, 1, 1, 1]),
                           "R": np.diag([1, 1]),
                           "q_xy": 1,
                           "q_psi": 1}

        # Model for optimization constraints
        self.model = model

        # Constants
        self.control_type = "NMPC"
        self.N = self.config["N"]    # Optimization horizon
        self.use_slack = use_slack

        # Spatial constraints
        self.space = space

    def update(self, parameters: dict) -> None:
        """
        Update model and optimization parameters

        Parameters
        ----------
            Parameters : dict
                Parameters = {
                    m_total, xg, Iz, Xudot, Yvdot, Nrdot,
                    Xu, Yv, Nr, k_port, k_stb, w1, w2, w3
                    theta_lambda, theta_V
                }


        """

        self.model.update(parameters)
        self.config["theta_lambda"] = parameters["theta_lambda"]
        self.config["theta_V"] = parameters["theta_V"]

    def step(self, x_init: np.ndarray, u_init: np.ndarray,
             x_desired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Steps controller

        Parameters
        ----------
        x_init : np.ndarray
            Initial state vector, i.e. [x, y, theta, u, v, r]
        u_init : np.ndarray
            Initial control signal
        x_desired : np.ndarray
            Goal state vector

        Returns
        -------
        x : np.ndarray
            State solution within the given horizon
        u : np.ndarray
            Optimal control vector within the given horizon
        """

        # Make optimization object
        opti = Optimizer()

        if False:
            # Ensure x_desired is the right shape
            if x_desired.shape[0] > 3:
                x_desired = x_desired.reshape(6, 1)
            else:
                x_desired = x_desired.reshape(3, 1)
            # Desired pose vector
            x_d = np.tile(x_desired, (1, self.N+1))  # .tolist()

            # Setup model specific optimization problem constraints
            if self.space:
                x, u, s = self.model.single_shooting(
                    x_init, u_init, opti, space=self.space, use_slack=self.use_slack)
            else:
                x, u, s = self.model.single_shooting(
                    x_init, u_init, opti, use_slack=self.use_slack)

            # Objective
            #
            if self.use_slack:
                opti.simple_quadratic(x, x_d, self.config, slack=s)
            else:
                opti.simple_quadratic(x, x_d, self.config)
            # opti.pseudo_huber(x, u, x_d, self.config)
            # opti.quadratic(x, u, x_d, self.config)

            # p_opts = {"expand": True}
            # s_opts = {"max_iter": 500, "print_level": 1, 'print_time': 0,
            #           'sb': 'yes', 'warm_start_init_point': 'yes'}
            # opti.solver("ipopt", p_opts,
            #             s_opts)

        if True:
            x, u, s = self.model.direct_collocation(x_init, u_init,
                                                    x_desired, self.config,
                                                    opti, self.space)

        # Use max iter?
        opts = {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.warm_start_init_point': 'yes'  # , "ipopt.max_iter": 500
        }
        opti.solver('ipopt', opts)

        # Setup solver and solve
        # opti.solver('ipopt')
        solution = opti.solve()

        # plot_solution(self.config["dt"], solution, x, u)
        return np.asarray(solution.value(x)), np.asarray(solution.value(u))

    def debug(self, x_init: np.ndarray, u_init: np.ndarray,
              x_desired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Steps controller with a try-except block and gives a crash report 
        if something goes wrong.

        Parameters
        ----------
        x_init : np.ndarray
            Initial state vector, i.e. [x, y, theta, u, v, r]
        u_init : np.ndarray
            Initial control signal
        x_desired : np.ndarray
            Goal state vector

        Returns
        -------
        x : np.ndarray
            State solution within the given horizon
        u : np.ndarray
            Optimal control vector within the given horizon
        """

        print(f"x_init.shape():     {x_init.shape}")
        print(f"x_desired.shape():  {x_desired.shape}")

        print(f"initial heading in cost: {x_init[2]}")
        print(f"desired heading in cost: {x_desired[2]}")

        # raise Exception("this is so dumb")

        if x_init.shape[0] != 6:
            raise Exception(f"Wrong initial state vector shape, \
            should be 6, got {x_init.shape[0]}")

        if x_desired.shape[0] != 3:
            raise Exception(f"Wrong initial state vector shape, \
            should be 3, got {x_desired.shape[0]}")

        crashed = False

        # Make optimization object
        opti = Optimizer()

        x, u, slack, J = self.model.direct_collocation(x_init, u_init,
                                                       x_desired, self.config,
                                                       opti, self.space)
        # x, u, slack = self.model.as_direct_collocation(x_init, u_init,
        #                                                x_desired, self.config,
        #                   '                             opti, self.space)

        # Use max iter?
        opts = {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.warm_start_init_point': 'yes'  # , "ipopt.max_iter": 500
        }
        opti.solver('ipopt', opts)

        try:
            solution = opti.solve()

            x_opt = np.asarray(solution.value(x))
            u_opt = np.asarray(solution.value(u))
            s_opt = np.asarray(solution.value(slack))

            delta = self.config["delta"]
            q_xy = self.config['q_xy']
            q_psi = self.config["q_psi"]
            Q = self.config["Q"]
            R = self.config["R"]
            # Find the different costs in a loop in order to find the real values,
            # and do the same in debug mode

            # Calculate cost
            linear = 0
            huber = 0
            heading = 0
            vel = 0
            act = 0
            # for n in range(self.N):
            #     linear += q_xy * utils.opt._pos_linear(x_opt[:, n], x_desired, delta)
            #     huber += q_xy * \
            #         utils.opt._pos_pseudo_huber(x_opt[:, n], x_desired, delta)
            #     heading += q_psi*utils.opt._heading_cost(x_opt, x_desired)
            #     vel += x_opt[3:6, n].T @ np.asarray(Q) @ x_opt[3:6, n]
            #     act += u_opt[:, n].T @ np.asarray(R) @ u_opt[:, n]

            # print(f"Huber cost: {huber}")
            # print(f"Head cost:  {heading}")
            # print(f"Vel cost:   {vel}")
            # print(f"Act cost:   {act}")
            # print(f"Total cost: {solution.value(J)}")

        except RuntimeError as error:
            crashed = True
            x_opt = None
            u_opt = np.zeros(2)
            s_opt = None

            print("===================================")
            print("---------- Crash report -----------")
            print(f"Error given: \n {error}")
            print(f"x: {np.round(opti.debug.value(x), 4)}")
            print(f"u: {np.round(opti.debug.value(u), 4)}")
            print(f"slack: {np.round(opti.debug.value(slack), 4)}")
            print(f"debug: {np.round(opti.debug.value(J), 4)}")
            print("===================================")

        return x_opt, u_opt, s_opt, crashed


class RLNMPC():
    def __init__(self, model: Model, dof: int = 2,
                 config: dict = None, space: tuple = None,
                 use_slack: bool = False) -> None:

        if config is None:
            self.config = {
                # "N": 50,
                "N": 10,
                "dt": 0.2,
                "Q": np.diag([1, 10, 50]).tolist(),
                "q_slack": [100, 100, 100, 100, 100, 100],
                "R": np.diag([0.01, 0.01]).tolist(),
                "delta": 10,
                "q_xy": 20,
                "q_psi": 100,
                "gamma": 0.99,
                "alpha": 0.01,  # RL Learning rate
                "beta": 0.01  # SYSID Learning rate
            }
        else:
            self.config = config

        self.space = space
        self.model = model
        self.Q_prev = 0
        self.V_value = 0
        theta_model = np.zeros(15)
        theta_initial = 0
        theta_terminal = np.zeros(3)
        self.theta = self.model.theta

        # Learning hyperparameters
        self.alpha = self.config["alpha"]
        self.beta = self.config["beta"]
        self.gamma = self.config["gamma"]

    def step(self, x_init: np.ndarray, u_init: np.ndarray, x_desired: np.ndarray) -> tuple:
        # TODO: Make this work like normal NMPC, when theta is the same as s
        Q_opti = Optimizer()

        # TODO: Use max iter?
        opts = {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            # , "ipopt.max_iter": 500
            "ipopt.linear_solver": "ma27",
            'ipopt.warm_start_init_point': 'yes', 'ipopt.tol_reached(x, x_d, pos_tol, head_tol)': True
        }

        print("Formulating Q step")
        x, u, s, theta, J_Q, grad, Lagrangian = self.model.Q_step(x_init, u_init,
                                                                  x_desired, self.theta, self.config,
                                                                  Q_opti, self.space)
        print("Solving")
        Q_opti.solver('ipopt', opts)
        Q = Q_opti.solve()

        # Find actual gradient value
        gradient = Q.value(grad)
        print(f"gradient: {gradient}")

        if self.Q_prev != 0:
            V_opti = Optimizer()

            print("Formulating V step")
            new_u, J_V = self.model.V_step(x_init, u_init,
                                           x_desired, self.theta, self.config,
                                           V_opti, self.space)

            print("Solving")
            V_opti.solver('ipopt', opts)
            V = V_opti.solve()
            V_current = V.value(J_V)  # Current state-function value

            # TD error expression, with objective function
            # from previous Q-function approximation
            yt = self.L_prev + self.gamma * V_current    # Target error estimate
            delta_expression = yt - self.J_prev

            # TD error value
            delta = yt - self.Q_prev.value(self.J_prev)  # TD error

            # Find hessian expression
            # Hessian is the transpose of the jacobian of the gradient
            # i.e. H(f) = J(grad_TD).T
            grad_TD = ca.gradient(delta_expression, self.theta_prev)
            hess_func = ca.Function("hess_func", [x, u, self.theta_prev],
                                    [ca.jacobian(grad_TD, self.theta_prev).T])
            hessian = hess_func(
                x_init,
                u_init,
                self.theta
            )

            try:
                hessian_inv = np.linalg.inv(hessian)
            except np.linalg.LinAlgError:
                hessian_inv = hessian

            # Update parameters theta using Q-learning
            self.theta += self.alpha*delta*self.gradient_prev   # Semi-gradient update

            if self.config["batch_size"] == 1:
                self.theta += self.alpha*delta*hessian_inv * \
                    self.gradient_prev   # Quasi-Newton update
            else:
                # Batch update
                ...

            self.theta = self.model.bound_theta(self.theta)
            print(f"Q_gradient: {self.gradient_prev}")
            print(f"Hessian:    {hessian}")
            print(f"theta:     {np.round(self.theta, 5)}")

            # self.model.update(self.theta)

        x_sol = np.asarray(Q.value(x))
        u_sol = np.asarray(Q.value(u))
        s_sol = np.asarray(Q.value(s))

        # Store information about previous states
        self.Lag_prev = Lagrangian
        self.prev_opti = Q_opti
        self.gradient_prev = gradient
        self.L_prev = utils.opt.pseudo_huber(x_sol[:, 0], u_sol[:, 0],
                                             x_desired, self.config, s_sol[:, 0])
        self.J_prev = J_Q
        self.theta_prev = theta

        return x_sol, u_sol


class GuidanceNMPC():
    def __init__(self, N: int = 100, config=None) -> None:
        self.N = N

        if config is None:
            pass
        else:
            self.config = config

    def step(self, x_init: np.ndarray, u_init: np.ndarray,
             x_desired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Make optimization object

        opti = Optimizer()

        guidance_model = OtterModel(dt=0.2, N=self.N)
        x, u, slack = guidance_model._init_opt(x_init, u_init, x_desired)
        T = opti.variable()

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

        # Start with an initial cost
        J = T
        dt = T/self.N

        # Formulate the NLP
        for k in range(self.N):
            # ===========================
            # State at collocation points
            # ===========================
            Xc = opti.variable(6, d)

            # TODO: Reformulate spatial constraints to
            #       include a safe boundary around the vessel
            # Spatial constraints
            # if space is not None:
            #     A, b = space
            #     for j in range(d):
            #         # State pos constraint
            #         opti.subject_to(A @ Xc[:2, j] <= b)

            opti.subject_to(opti.bounded(utils.kts2ms(-5),
                                         Xc[3:5, :],
                                         utils.kts2ms(5)))
            opti.subject_to(opti.bounded(-np.pi, Xc[5, :], np.pi))

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
                # fj = self.step(Xc[:, j-1], u[:, k])
                implicit = guidance_model.implicit(Xc[:, j-1], u[:, k], xp, dt)

                # Collocation objective function contribution
                # qj = utils.opt.pseudo_huber(
                #     Xc[:, j-1], u[:, k], x_desired, self.config, slack[:, k])

                # Apply dynamics with forward euler
                # this is where the dynamics integration happens
                # opti.subject_to(self.dt*fj == xp)
                opti.subject_to(implicit == 0)

                # Add contribution to the end state
                Xk_end = Xk_end + D[j]*Xc[:, j-1]

                # Add contribution to quadrature function using forward euler
                # this is where the objective integration happens
                # J = J + B[j]*qj*dt

            # Add equality constraint
            opti.subject_to(x[:, k+1] == Xk_end)

        # Minimize objective
        opti.minimize(J)

        # Use max iter?
        opts = {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.warm_start_init_point': 'yes'  # , "ipopt.max_iter": 500
        }
        opti.solver('ipopt', opts)

        # Setup solver and solve
        # opti.solver('ipopt')
        solution = opti.solve()

        # plot_solution(self.config["dt"], solution, x, u)
        return np.asarray(solution.value(x)), np.asarray(solution.value(u))
