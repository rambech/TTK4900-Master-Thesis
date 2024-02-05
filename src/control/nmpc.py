import numpy as np
from .control import Control
from .objectives import time
from vehicle.models import Model, DubinsCarModel
import casadi as ca
import matplotlib.pyplot as plt
from utils import D2R


class nmpc(Control):
    """
    Nonlinear Model Predictive Control class
    """

    # Constants
    control_type = "NMPC"

    def __init__(self, model: Model, dof: int = 2, horizon: int = 40) -> None:
        super().__init__(dof)
        self.N = horizon  # Optimization horizon

    def step(self, x, prev_u) -> np.ndarray:
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
    return ca.vertcat(x[1], - g/l * ca.sin(x[0]) + u)


def pendulum_example():
    """
    Temporary procedure for testing casadi functionality
    """
    # Time horizon
    N = 200

    opti = ca.Opti()
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


def dubins_time_example():
    mpc_model = DubinsCarModel()
    N = 200  # Time horizon

    # Making optimization object
    opti = ca.Opti()

    # Declaring optimization variables
    x = opti.variable(3, N+1)
    x_pos = x[0, :]
    y_pos = x[1, :]
    theta = x[2, :]
    u = opti.variable(2, N)
    v = u[0, :]
    phi = u[1, :]
    # T = opti.variable()

    # Objective
    # opti.minimize(T)

    # # Time step
    # dt = T/N
    dt = time(N, opti)

    # Fixed step Runge-Kutta 4 integrator
    for k in range(N):
        k1 = mpc_model.step(x[:, k],             u[:, k])
        k2 = mpc_model.step(x[:, k] + dt/2 * k1, u[:, k])
        k3 = mpc_model.step(x[:, k] + dt/2 * k2, u[:, k])
        k4 = mpc_model.step(x[:, k] + dt * k3,   u[:, k])
        x_next = x[:, k] + dt/6 * (k1+2*k2+2*k3+k4)
        opti.subject_to(x[:, k+1] == x_next)

    # Control signal and time constraint
    opti.subject_to(opti.bounded(0, v, 1))
    opti.subject_to(opti.bounded(D2R(-15), phi, D2R(15)))
    # opti.subject_to(T >= 0)

    # Boundary values
    # Initial conditions
    opti.subject_to(x_pos[0] == 0)
    opti.subject_to(y_pos[0] == 0)
    opti.subject_to(theta[0] == 0)
    opti.set_initial(x_pos, 0)
    opti.set_initial(y_pos, 0)
    opti.set_initial(theta, 0)
    # opti.set_initial(T, 1)

    # End conditions
    opti.subject_to(x_pos[-1] == 10)
    opti.subject_to(y_pos[-1] == 0.5)
    opti.subject_to(theta[-1] == np.pi/2)

    p_opts = {"expand": True}
    s_opts = {"max_iter": 100}
    opti.solver("ipopt", p_opts,
                s_opts)

    # Setup solver and solve
    opti.solver('ipopt')
    solution = opti.solve()

    print(f"x_pos_opt {solution.value(x_pos)}")
    print(f"y_pos_opt {solution.value(y_pos)}")
    # print(solution.value(T))

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_aspect("equal")
    ax.plot(solution.value(x_pos), solution.value(y_pos))

    plt.show()


def dubins_distance_example():
    mpc_model = DubinsCarModel()
    N = 40  # Time horizon

    # Making optimization object
    opti = ca.Opti()

    # Declaring optimization variables
    x = opti.variable(3, N+1)
    x_pos = x[0, :]
    y_pos = x[1, :]
    theta = x[2, :]
    u = opti.variable(2, N)
    v = u[0, :]
    phi = u[1, :]
    # T = opti.variable()

    # Objective
    # opti.minimize(T)

    # Time step
    dt = 0.05

    # Fixed step Runge-Kutta 4 integrator
    for k in range(N):
        k1 = mpc_model.step(x[:, k],             u[:, k])
        k2 = mpc_model.step(x[:, k] + dt/2 * k1, u[:, k])
        k3 = mpc_model.step(x[:, k] + dt/2 * k2, u[:, k])
        k4 = mpc_model.step(x[:, k] + dt * k3,   u[:, k])
        x_next = x[:, k] + dt/6 * (k1+2*k2+2*k3+k4)
        opti.subject_to(x[:, k+1] == x_next)

    # Control signal and time constraint
    opti.subject_to(opti.bounded(0, v, 1))
    opti.subject_to(opti.bounded(D2R(-15), phi, D2R(15)))
    # opti.subject_to(T >= 0)

    # Boundary values
    # Initial conditions
    opti.subject_to(x_pos[0] == 0)
    opti.subject_to(y_pos[0] == 0)
    opti.subject_to(theta[0] == 0)
    opti.set_initial(x_pos, 0)
    opti.set_initial(y_pos, 0)
    opti.set_initial(theta, 0)
    # opti.set_initial(T, 1)

    # End conditions
    opti.subject_to(x_pos[-1] == 10)
    opti.subject_to(y_pos[-1] == 0.5)
    opti.subject_to(theta[-1] == np.pi/2)

    # Setup solver and solve
    opti.solver('ipopt')
    solution = opti.solve()

    print(f"x_pos_opt {solution.value(x_pos)}")
    print(f"y_pos_opt {solution.value(y_pos)}")
    print(solution.value(T))

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_aspect("equal")
    ax.plot(solution.value(x_pos), solution.value(y_pos))

    plt.show()
