import numpy as np
from vehicle.models import Model
from casadi import Opti, sin
import matplotlib.pyplot as plt


class nmpc():
    def __init__(self, model: Model, horizon: int = 40) -> None:
        self.N = horizon  # Optimization horizon

    def step(self, eta, nu) -> np.ndarray:
        """
        Steps nmpc controller

        Parameters 
        ----------
        eta : np.ndarray
            Current pose vector in 3 DOF
        nu : np.ndarray
            Current velocity vector in 3 DOF

        Returns
        -------
        u : np.ndarray
            Optimal control vector within the given horizon
        """
        pass


def test_casadi():
    """
    Temporary procedure for testing casadi functionality
    """
    opti = Opti()
    x = opti.variable(2)
    u = opti.variable()  # Torque
    g = opti.parameter()
    l = opti.parameter()
    opti.set_value()
    opti.set_value(g, -9.81)
    opti.set_value(l, 1)

    opti.minimize(x[0])
    opti.subject_to(x[1]+g/l * sin(x[0]) - u == 0)
    opti.subject_to(opti.bounded(-3*np.pi, x[0], 3*np.pi))
    opti.subject_to(opti.bounded(-np.pi, x[1], np.pi))
    opti.subject_to(opti.bounded(-1, u, 1))

    p_opts = {"expand": True}
    s_opts = {"max_iter": 100}
    opti.solver("ipopt", p_opts,
                s_opts)

    # opti.set_initial(x[0], np.pi/2)
    # opti.set_initial(x[1], 0)
    solution = opti.solve()

    print(f"x: {solution.value(x[0])}")
    print(f"u: {solution.value(u)}")


test_casadi()
