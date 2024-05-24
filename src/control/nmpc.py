import numpy as np

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

        # Learning hyperparameters
        self.alpha = self.config["alpha"]
        self.beta = self.config["beta"]
        self.gamma = self.config["gamma"]

    def step(self, x_init: np.ndarray, u_init: np.ndarray, x_desired: np.ndarray) -> tuple:
        # TODO: Make this work like normal NMPC, when theta is the same as s
        Q_opti = Optimizer()

        # TODO: Use max iter?
        # opts = {
        #     'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
        #     # , "ipopt.max_iter": 500
        #     "ipopt.linear_solver": "ma27",
        #     'ipopt.warm_start_init_point': 'yes'
        #     # , 'ipopt.tol_reached': True
        # }
        opts = {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.warm_start_init_point': 'yes'  # , "ipopt.max_iter": 500
        }

        print("Formulating Q step")
        x, u, s, theta, Q, grad, grad_f, Lagrangian = self.model.Q_step(x_init, u_init,
                                                                        x_desired, self.theta, self.config,
                                                                        Q_opti, self.space)
        # TODO: Save cost values
        print("Solving")
        Q_opti.solver('ipopt', opts)
        Q_solution = Q_opti.solve()

        # Find actual gradient value
        gradient = Q_solution.value(grad)
        print(f"Q_gradient: {gradient}")

        # TODO: Is it right that this is a jacobian?
        gradient_f = Q_solution.value(grad_f)
        # print(f"gradient_f: {gradient_f}")

        if self.Q_prev != 0 and (self.alpha > 0.0 or self.beta > 0.0):
            V_opti = Optimizer()

            print("Formulating V step")
            policy, V, policy_x, policy_u, p_ss = self.model.V_step(x_init, u_init,
                                                                    x_desired, self.theta, self.config,
                                                                    V_opti, self.space)

            print("Solving")
            V_opti.solver('ipopt', opts)
            V_solution = V_opti.solve()
            V_current = V_solution.value(V)  # Current state-function value

            # Target error estimate
            yt = self.L_prev + self.gamma * V_current
            print(f"TD target: {yt}")

            # TD error value
            delta = yt - self.Q_prev
            print(f"delta: {delta}")

            # Model prediction error x_t - f(x_t-1, u_t-1)
            e = x_init - self.x_prev
            e = e.reshape((e.shape[0], 1))
            print(f"Prediction error: {e}")

            # Update parameters theta using Q-learning
            if self.config["batch size"] > 0:
                # TODO: Fix, cost parameters not updateing
                # Use batch size if specified

                # If batch size == 1, the update becomes quasi-Newton
                if self.batch_count == 1:
                    self.J_Q = np.array([self.gradient_prev])
                    # print(f"self.gradient_prev: {self.gradient_prev}")
                    # print(f"self.J_Q: {self.J_Q}")
                    self.J_f = self.x_grad_prev
                    # print(f"self.x_grad_prev: {self.x_grad_prev}")
                    # print(f"self.J_f: {self.J_f}")
                    # print(
                    #     f"self.x_grad_prev.shape: {self.x_grad_prev.shape}")
                    # print(f"self.J_f.shape: {self.J_f.shape}")
                    self.deltas = np.array([delta])
                    self.es = e.reshape((e.shape[0], 1))
                    # print(f"self.es.shape: {self.es.shape}")

                else:
                    self.J_Q = np.vstack((self.J_Q, self.gradient_prev))
                    # print(f"self.gradient_prev: {self.gradient_prev}")
                    # print(f"self.J_Q: {self.J_Q}")
                    # self.J_f = np.vstack((self.J_f, self.x_grad_prev))
                    # print(f"self.J_f.shape: {self.J_f.shape}")
                    # print(f"self.J_f: {self.J_f}")
                    # print(f"self.x_grad_prev.shape: {self.x_grad_prev.shape}")
                    # self.J_f = np.concatenate(
                    #     (self.J_f, self.x_grad_prev), axis=0)
                    self.J_f = sparse.vstack((self.J_f, self.x_grad_prev))
                    self.deltas = np.vstack((self.deltas, delta))
                    # print(f"self.deltas: {self.deltas}")
                    # print(f"self.J_f.shape: {self.J_f.shape}")
                    # print(f"self.J_f: {self.J_f}")
                    # print(f"self.x_grad_prev.shape: {self.x_grad_prev.shape}")
                    self.es = np.vstack((self.es, e))
                    # print(f"self.es.shape: {self.es.shape}")

                if self.config["batch size"] == self.batch_count:
                    # =======================
                    # RL Gauss-Newton update
                    # =======================
                    Q_hessian = (
                        self.J_Q.T @ self.J_Q +
                        self.config["lq"]*np.eye(self.J_Q.shape[1])
                    )
                    # print(f"Q_hessian.shape: {Q_hessian.shape}")
                    # print(f"self.J_Q.shape: {self.J_Q.shape}")
                    Q_hessian_inv = np.linalg.inv(Q_hessian)
                    # print(f"Q_hessian_inv: {Q_hessian_inv}")
                    # print(f"self.J_Q: {self.J_Q}")
                    temp = Q_hessian_inv.dot(self.J_Q.T)
                    nabla_Q = temp @ self.deltas
                    nabla_Q = nabla_Q.reshape((nabla_Q.shape[0], 1))
                    # print(f"temp.shape: {temp.shape}")
                    # print(f"self.deltas.shape: {self.deltas.shape}")
                    # print(f"nabla_Q.shape: {nabla_Q.shape}")
                    # print(f"self.theta.shape: {self.theta.shape}")

                    # =======================
                    # PEM Gauss-Newton update
                    # =======================
                    # print(f"J_f.T.shape: {self.J_f.T.shape}")
                    # print(f"J_f.shape: {self.J_f.shape}")
                    # print(
                    #     f"self.es.shape: {self.es.shape}")
                    f_hessian = (
                        self.J_f.T @ self.J_f +
                        self.config["lf"]*np.eye(self.J_f.shape[1])
                    )
                    f_hessian_inv = np.linalg.inv(f_hessian)
                    # print(f"f_hessian_inv.shape: {f_hessian_inv.shape}")
                    new_temp = f_hessian_inv @ self.J_f.T
                    # print(f"new_temp.shape: {new_temp.shape}")
                    nabla_f = new_temp @ self.es
                    # print(
                    #     f"nabla_f.shape: {nabla_f.shape}")
                    # nabla_f = nabla_f.reshape(nabla_f.shape[0],)
                    # print(
                    #     f"nabla_f.shape: {nabla_f.shape}")

                    # =======================
                    # Update theta
                    # =======================
                    if self.config["projection threshold"] > 0:
                        U, S, Vh = np.linalg.svd(Q_hessian)
                        # V = Vh.T
                        # TODO: Choose all rows of Vh.T that are below projection threshold
                        # print(f"Vh: {Vh}")
                        print(f"S: {S}")
                        mask = S < self.config["projection threshold"]
                        # print(f"flattend: {flat}")
                        print(f"mask: {mask}")
                        if np.any(mask):
                            Vh_reduced = Vh[mask]
                            proj = Vh_reduced.T @ Vh_reduced
                            # print(f"Vh: {Vh}")
                            # print(f"Vh_reduced: {Vh_reduced}")
                            print(f"proj: {proj}")
                            print(f"nabla_f: {nabla_f}")
                            print(f"proj @ nabla_f: {proj @ nabla_f}")
                            print(f"Vh.shape: {Vh.shape}")
                            print(f"Vh_reduced.shape: {Vh_reduced.shape}")
                            print(f"proj.shape: {proj.shape}")
                            print(f"nabla_f.shape: {nabla_f.shape}")
                            print(
                                f"proj @ nabla_f.shape: {(proj @ nabla_f).shape}")
                            self.theta += self.alpha*nabla_Q + self.beta * proj @ nabla_f
                        else:
                            self.theta += self.alpha*nabla_Q + self.beta*nabla_f
                    else:
                        self.theta += self.alpha*nabla_Q + self.beta*nabla_f
                    # TODO: Make SYSID work
                    # print(f"self.theta.shape: {self.theta.shape}")
                    # U, S, Vh = np.linalg.svd(Q_hessian)
                    # print(f"Vh: {Vh}")
                    # print(f"S: {S}")
                    # print(f"Vh.shape: {Vh.shape}")
                    # print(f"S.shape: {S.shape}")

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

            print(f"Q_gradient: {self.gradient_prev}")
            print(f"theta:     {np.round(self.theta, 5)}")
            print(f"Original theta:     {np.round(self.init_theta, 5)}")
            print(
                f"Diff theta:     {np.round(self.theta - self.init_theta, 5)}")

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
        self.L_prev = utils.opt.np_pseudo_huber(x_sol[:, 0], u_sol[:, 0],
                                                x_desired, self.config, s_sol[:, 0])

        return x_sol, u_sol, s_sol, self.theta

    def debug(self, x_init: np.ndarray, u_init: np.ndarray,
              x_desired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        try:
            x_sol, u_sol, s_sol, theta = self.step(x_init, u_init, x_desired)
            error_caught = False

        except RuntimeError as error:
            print(f"Error: {error}")
            x_sol, u_sol, s_sol = None, np.zeros(2), None
            error_caught = True
            theta = None

        return x_sol, u_sol, s_sol, theta, error_caught
