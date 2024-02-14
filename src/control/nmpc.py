import numpy as np
from .control import Control
from .optimizer import Optimizer
from vehicle.models import Model
import casadi as ca
from plotting import plot_solution


class NMPC(Control):
    """
    Nonlinear Model Predictive Control class
    """

    def __init__(self, model: Model, dof: int = 2, horizon: int = 40, dt: float = 0.05) -> None:
        """
        Parameters
        ----------
        model : Model
            Model for OCP constraints
        dof : int
            Degrees of freedom
        horizon : int
            Optimization horizon
        dt : float
        """

        super().__init__(dof)
        self.N = horizon    # Optimization horizon
        self.model = model  # Model

        # Constants
        self.control_type = "NMPC"

    def step(self, x_init, u_init, x_desired) -> tuple[np.ndarray, np.ndarray]:
        """
        Steps NMPC controller

        Parameters 
        ----------
        x_init : np.ndarray
            Initial state vector
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

        # Desired pose vector
        x_desired = np.tile(x_desired, (self.N+1, 1)).tolist()
        x_d = ca.hcat(x_desired)

        # Setup model specific optimization problem constraints
        x, u, s = self.model.setup_opt(x_init, u_init, opti)

        # Objective
        opti.simple_quadratic(x, x_d)
        # opti.quadratic(x, u, x_d)

        # Setup solver and solve
        opti.solver('ipopt')
        solution = opti.solve()

        plot_solution(solution, x, u)
        return solution.value(x), solution.value(u)
