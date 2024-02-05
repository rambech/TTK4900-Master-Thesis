import numpy as np
from .control import Control
from vehicle.models import Model
from casadi import Opti, sin, vertcat
import matplotlib.pyplot as plt


class nmpc(Control):
    """
    Nonlinear Model Predictive Control class
    """

    # Constants
    control_type = "NMPC"

    def __init__(self, model: Model, dof: int = 2, horizon: int = 40) -> None:
        super().__init__(dof)
        self.N = horizon  # Optimization horizon

    def step(self, eta, nu, prev_u) -> np.ndarray:
        """
        Steps NMPC controller

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


def pendulum_dynamics(x, u):
    g = 9.81
    l = 1
    return vertcat(x[1], - g/l * sin(x[0]) + u)


def dumb(x):
    g = 9.81
    l = 1
    return vertcat(x[1], - g/l * sin(x[0]))


def pendulum_example():
    """
    Temporary procedure for testing casadi functionality
    """
    # Time horizon
    N = 200

    opti = Opti()
    x = opti.variable(2, N+1)
    theta = x[0, :]
    theta_dot = x[1, :]
    u = opti.variable(N)  # Torque
    T = opti.variable()

    opti.minimize(T)

    dt = T/N

    # RK4
    for i in range(N):
        k1 = pendulum_dynamics(x[:, i],           u[i])
        k2 = pendulum_dynamics(x[:, i] + dt*k1/2, u[i])
        k3 = pendulum_dynamics(x[:, i] + dt*k2/2, u[i])
        k4 = pendulum_dynamics(x[:, i] + dt*k3,   u[i])

        # k1 = dumb(x[:, i])
        # k2 = dumb(x[:, i] + dt*k1/2)
        # k3 = dumb(x[:, i] + dt*k2/2)
        # k4 = dumb(x[:, i] + dt*k3)

        x_next = x[:, i] + dt/6 * (k1+2*k2+2*k3+k4)

        opti.subject_to(x[:, i+1] == x_next)

    # opti.subject_to(opti.bounded(-3*np.pi, theta, 3*np.pi))
    # opti.subject_to(opti.bounded(-np.pi, theta_dot, np.pi))

    # Control signal constraint
    opti.subject_to(opti.bounded(-0.2, u, 0.2))
    opti.subject_to(T >= 0)

    # Boundary conditions
    # Initial conditions
    opti.subject_to(theta[0] == np.pi/2)
    # opti.subject_to(theta_dot[0] == 0)
    opti.set_initial(theta, 0)
    opti.set_initial(theta_dot, 0)
    opti.set_initial(T, 1)

    # End conditions
    opti.subject_to(theta[-1] == np.pi)
    # opti.subject_to(theta[-1] == 0)
    opti.subject_to(theta_dot[-1] == 0)

    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000}
    opti.solver("ipopt", p_opts,
                s_opts)

    solution = opti.solve()
    # print(f"Debug theta: {opti.debug.value(theta)}")

    print(f"theta: {solution.value(theta)}")
    print(f"theta_dot: {solution.value(theta_dot)}")
    # print(f"u: {solution.value(u)}")
    print(solution.value(T))

    fig, ax = plt.subplots(figsize=(7, 7))
    alpha = -np.pi/2
    x_pos = np.cos(solution.value(theta)+alpha)
    y_pos = np.sin(solution.value(theta)+alpha)

    # print(f"x pos: {x_pos}")
    # print(f"y pos: {y_pos}")

    ax.set_aspect("equal")
    ax.plot(x_pos, y_pos)

    plt.show()


def dubins_example():
    ...

# test_casadi()
