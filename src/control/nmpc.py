import numpy as np

import utils.linalg
import utils.opt
from .control import Control
from .optimizer import Optimizer
from vehicle.models import Model, OtterModel
import casadi as ca
from plotting import plot_solution
import utils
from scipy import sparse

# TODO: Determine the minimum controller speed needed,
#       in order to control the Otter


class NMPC(Control):
    """
    Nonlinear Model Predictive Control class
    """

    def __init__(self, model: Model, dof: int = 2,
                 config: dict = None, space: tuple = None,
                 use_slack: bool = False, plan_count: int = 10) -> None:
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
        self.plan_count = plan_count

        # Spatial constraints
        self.space = space

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
    types = ("setpoint", "tracking", "planning")

    def __init__(self, model: Model, config: dict = None, type: str = "setpoint",
                 space: tuple = None, use_slack: bool = False, plan_count: int = 10) -> None:

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

        self.N = self.config["N"]

        self.space = space
        self.model = model
        self.Q_prev = 0
        self.V_value = 0
        theta = self.model.theta
        self.theta = theta.reshape(theta.shape[0], 1)
        self.init_theta = self.theta.copy()
        self.batch_count = 1
        self.J_Q = 0
        self.J_f = 0
        self.deltas = 0
        self.es = 0
        self.x_prev = 0
        self.x_grad_prev = 0
        if "plan count" in self.config:
            self.plan_count = self.config["plan count"]

        if type not in (self.types):
            raise Exception(f"{type} is not a valid mpc type")

        self.type = type
        self.warm_start = None
        self.V_warm_start = None

        # Learning hyperparameters
        self.alpha = self.config["alpha"]
        self.beta = self.config["beta"]
        self.gamma = self.config["gamma"]

    def step(self, x_init: np.ndarray, u_init: np.ndarray, x_desired: np.ndarray) -> tuple:
        Q_opti = Optimizer()

        opts = {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.warm_start_init_point': 'yes'  # , "ipopt.max_iter": 500
        }

        print("Formulating Q step")
        x, u, s, theta, Q, grad, grad_f, Lagrangian = self.model.rl_step(x_init, u_init,
                                                                         x_desired, self.theta,
                                                                         self.config, Q_opti,
                                                                         self.space, step_type="Q",
                                                                         warm_start=self.warm_start)
        # self.warm_start = None
        # self.V_warm_start = None

        if self.warm_start is not None:
            x_ws, u_ws, lam_g = self.warm_start
            Q_opti.set_initial(x, x_ws)
            Q_opti.set_initial(u, u_ws)
            Q_opti.set_initial(Q_opti.lam_g, lam_g)

        # TODO: Save cost values
        print("Solving")
        Q_opti.solver('ipopt', opts)
        Q_solution = Q_opti.solve()

        # Find actual gradient value
        gradient = Q_solution.value(grad)

        gradient_f = Q_solution.value(grad_f)

        if self.Q_prev != 0 and (self.alpha > 0.0 or self.beta > 0.0):
            V_opti = Optimizer()

            print("Formulating V step")
            policy_x, policy, p_ss, V = self.model.rl_step(x_init, u_init,
                                                           x_desired, self.theta,
                                                           self.config, V_opti,
                                                           self.space, step_type="V",
                                                           warm_start=self.V_warm_start)
            # V_opti.set_initial(self.warm_start)
            if self.V_warm_start is not None:
                x_ws, u_ws, lam_g = self.V_warm_start
                V_opti.set_initial(policy_x, x_ws)
                V_opti.set_initial(policy, u_ws)
                V_opti.set_initial(V_opti.lam_g, lam_g)

            print("Solving")
            V_opti.solver('ipopt', opts)
            V_solution = V_opti.solve()
            V_current = V_solution.value(V)  # Current state-function value
            self.V_warm_start = (
                V_solution.value(policy_x), V_solution.value(policy),
                V_solution.value(V_opti.lam_g)
            )

            # Target error estimate
            yt = self.L_prev + self.gamma * V_current

            # TD error value
            delta = yt - self.Q_prev

            # Prediction error
            e = x_init - self.x_prev
            e = e.reshape((e.shape[0], 1))

            # Update parameters theta using Q-learning
            if self.config["batch size"] > 0:
                # Use batch size if specified

                # If batch size == 1, the update becomes quasi-Newton
                if self.batch_count == 1:
                    self.J_Q = np.array([self.gradient_prev])
                    self.J_f = self.x_grad_prev
                    self.deltas = np.array([delta])
                    self.es = e.reshape((e.shape[0], 1))

                else:
                    self.J_Q = np.vstack((self.J_Q, self.gradient_prev))
                    self.J_f = sparse.vstack((self.J_f, self.x_grad_prev))
                    self.deltas = np.vstack((self.deltas, delta))
                    self.es = np.vstack((self.es, e))

                if self.config["batch size"] == self.batch_count:
                    # =======================
                    # RL Gauss-Newton update
                    # =======================
                    Q_hessian = (
                        self.J_Q.T @ self.J_Q +
                        self.config["lq"]*np.eye(self.J_Q.shape[1])
                    )
                    Q_hessian_inv = np.linalg.inv(Q_hessian)
                    temp = Q_hessian_inv.dot(self.J_Q.T)
                    nabla_Q = temp @ self.deltas
                    nabla_Q = nabla_Q.reshape((nabla_Q.shape[0], 1))

                    # =======================
                    # PEM Gauss-Newton update
                    # =======================
                    f_hessian = (
                        self.J_f.T @ self.J_f +
                        self.config["lf"]*np.eye(self.J_f.shape[1])
                    )
                    f_hessian_inv = np.linalg.inv(f_hessian)
                    new_temp = f_hessian_inv @ self.J_f.T
                    nabla_f = new_temp @ self.es

                    # =======================
                    # Update theta
                    # =======================
                    if self.config["projection threshold"] > 0:
                        proj = utils.linalg.singular_projection(
                            Q_hessian, self.config["projection threshold"])

                        self.theta += self.alpha*nabla_Q + self.beta * proj @ nabla_f
                    else:
                        self.theta += self.alpha*nabla_Q + self.beta*nabla_f

                    # Reset quatities
                    self.batch_count = 1
                    self.J_Q = 0
                    self.J_f = 0
                    self.deltas = 0
                    self.es = 0

                else:
                    self.batch_count += 1

            else:
                # Use semi-gradient update if batch size not specified
                self.theta += self.alpha*delta*self.gradient_prev

            # print(f"theta:     {np.round(self.theta, 5)}")
            # print(f"Original theta:     {np.round(self.init_theta, 5)}")
            # print(
            #     f"Diff theta:     {np.round(self.theta - self.init_theta, 5)}")

        x_sol = np.asarray(Q_solution.value(x))
        u_sol = np.asarray(Q_solution.value(u))
        s_sol = np.asarray(Q_solution.value(s))

        # Store information about previous states
        self.x_prev = x_sol[:, 1]
        self.x_grad_prev = gradient_f
        self.Lag_prev = Lagrangian
        self.prev_opti = Q_opti
        self.Q_prev = Q_opti.value(Q)
        self.gradient_prev = gradient
        self.warm_start = (x_sol, u_sol,
                           Q_solution.value(Q_opti.lam_g))

        if x_desired.ndim == 2:
            self.L_prev = utils.opt.np_pseudo_huber(x_sol[:, 0], u_sol[:, 0],
                                                    x_desired[:3, 0], self.config, s_sol[:, 0])
        else:
            self.L_prev = utils.opt.np_pseudo_huber(x_sol[:, 0], u_sol[:, 0],
                                                    x_desired, self.config, s_sol[:, 0])

        return x_sol, u_sol, s_sol, self.theta, self.Q_prev, self.L_prev

    def plan_step(self, x_init: np.ndarray, u_init: np.ndarray, x_desired: np.ndarray) -> tuple:
        opti = Optimizer()

        opts = {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.warm_start_init_point': 'yes'  # , "ipopt.max_iter": 500
        }

        print("Formulating plannning step")
        x, u, s, _, _, _, _, _ = self.model.rl_step(x_init, u_init,
                                                    x_desired, self.theta, self.config,
                                                    opti, self.space, step_type="Q")
        # TODO: Save cost values
        print("Solving")
        opti.solver('ipopt', opts)
        solution = opti.solve()

        x_plan = np.asarray(solution.value(x))

        return x_plan

    def debug(self, x_init: np.ndarray, u_init: np.ndarray,
              x_desired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        if self.type in ("setpoint", "tracking"):
            try:
                x_sol, u_sol, s_sol, theta, cost, stage_cost = self.step(
                    x_init, u_init, x_desired)
                error_caught = False

            except RuntimeError as error:
                print(f"Error: {error}")
                x_sol, u_sol, s_sol = None, np.zeros(2), None
                error_caught = True
                theta = None
                cost = None
                stage_cost = None

            return x_sol, u_sol, s_sol, theta, cost, stage_cost, error_caught

        elif self.type == "planning":
            try:
                x_plan = self.plan_step(
                    x_init, u_init, x_desired)
                error_caught = False

            except RuntimeError as error:
                print(f"Error: {error}")
                x_plan = np.zeros(6)
                error_caught = True

            return x_plan, error_caught
