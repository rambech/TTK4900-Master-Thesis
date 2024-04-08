import numpy as np
from .control import Control
from .optimizer import Optimizer
from vehicle.models.models import Model
import casadi as ca
from plotting import plot_solution

# TODO: Add configuration dictionary for more flexible testing


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

        # Use max iter?
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
                'ipopt.warm_start_init_point': 'yes', "ipopt.max_iter": 500}
        opti.solver('ipopt', opts)

        # Setup solver and solve
        # opti.solver('ipopt')
        solution = opti.solve()

        # plot_solution(self.config["dt"], solution, x, u)
        return np.asarray(solution.value(x)), np.asarray(solution.value(u))
