import numpy as np

import utils.opt
from .control import Control
from .optimizer import Optimizer
from vehicle.models.models import Model
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
                "Q_slack": np.diag([100, 100, 100, 100, 100, 100]).tolist(),
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
        Q_opti = Optimizer()

        # TODO: Use max iter?
        opts = {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.warm_start_init_point': 'yes'  # , "ipopt.max_iter": 500
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
            # self.theta += self.alpha*delta*hessian_inv*Q_gradient   # Quasi-Newton update

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
