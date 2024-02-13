import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from .nmpc import NMPC
from .optimizer import Optimizer
from vehicle.models import DubinsCarModel
from utils import D2R
from plotting import plot_solution


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
    opti.set_initial(theta, 0)
    opti.set_initial(theta_dot, 0)
    opti.set_initial(T, 1)

    # End conditions
    opti.subject_to(theta[-1] == np.pi)
    opti.subject_to(theta_dot[-1] == 0)

    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000}
    opti.solver("ipopt", p_opts,
                s_opts)

    solution = opti.solve()

    print(f"theta: {solution.value(theta)}")
    print(f"theta_dot: {solution.value(theta_dot)}")
    print(solution.value(T))

    fig, ax = plt.subplots(figsize=(7, 7))
    alpha = -np.pi/2
    x_pos = np.cos(solution.value(theta)+alpha)
    y_pos = np.sin(solution.value(theta)+alpha)

    ax.set_aspect("equal")
    ax.plot(x_pos, y_pos)

    plt.show()


def dubins_time_example():
    mpc_model = DubinsCarModel()
    N = 200  # Time horizon

    # Making optimization object
    opti = Optimizer()

    # Declaring optimization variables
    x = opti.variable(3, N+1)
    x_pos = x[0, :]
    y_pos = x[1, :]
    theta = x[2, :]
    u = opti.variable(2, N)
    v = u[0, :]
    phi = u[1, :]

    # Objective
    dt = opti.time(N)

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

    # Boundary values
    # Initial conditions
    opti.subject_to(x_pos[0] == 0)
    opti.subject_to(y_pos[0] == 0)
    opti.subject_to(theta[0] == 0)
    opti.set_initial(x_pos, 0)
    opti.set_initial(y_pos, 0)
    opti.set_initial(theta, 0)

    # End conditions
    opti.subject_to(x_pos[-1] == 0.25)
    opti.subject_to(y_pos[-1] == 0.25)
    opti.subject_to(theta[-1] == np.pi)

    p_opts = {"expand": True}
    s_opts = {"max_iter": 100}
    opti.solver("ipopt", p_opts,
                s_opts)

    # Setup solver and solve
    opti.solver('ipopt')
    solution = opti.solve()

    print(f"x_pos_opt {solution.value(x_pos)}")
    print(f"y_pos_opt {solution.value(y_pos)}")

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_aspect("equal")
    ax.plot(solution.value(x_pos), solution.value(y_pos))

    plt.show()


def dubins_distance_example():
    mpc_model = DubinsCarModel()
    N = 300  # Time horizon

    # Making optimization object
    opti = Optimizer()

    # Declaring optimization variables
    # State variables
    x = opti.variable(3, N+1)
    x_pos = x[0, :]
    y_pos = x[1, :]
    theta = x[2, :]

    # Input variables
    u = opti.variable(2, N)
    v = u[0, :]
    phi = u[1, :]

    # Slack variables
    s = opti.variable(3, N+1)

    x_desired = np.tile([10, 10, 0], (N+1, 1)).tolist()
    x_d = ca.hcat(x_desired)

    # Objective
    opti.quadratic(x, x_d)

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

    # Boundary values
    # Initial conditions
    opti.subject_to(x_pos[0] == 0)
    opti.subject_to(y_pos[0] == 0)
    opti.subject_to(theta[0] == 0)
    opti.set_initial(x_pos, 0)
    opti.set_initial(y_pos, 0)
    opti.set_initial(theta, 0)

    # Setup solver and solve
    opti.solver('ipopt')
    solution = opti.solve()

    print(f"x_pos_opt {solution.value(x_pos)}")
    print(f"y_pos_opt {solution.value(y_pos)}")

    plot_solution(solution, x, u)


def new_distance_example():
    dt = 0.05
    N = 300  # Time horizon

    # Making optimization object
    opti = Optimizer()

    u_init = np.zeros(2)
    x_init = np.zeros(3)
    x_d = ca.hcat([10, 10, 0])

    mpc_model = DubinsCarModel(dt=dt, N=N)
    x, u, s = mpc_model.setup_opt(x_init, u_init, opti)

    # Objective
    opti.quadratic(x, x_d)

    # Setup solver and solve
    opti.solver('ipopt')
    solution = opti.solve()

    plot_solution(solution, x, u)


def test_mpc():
    dt = 0.05
    N = 40
    mpc_model = DubinsCarModel()
    controller = NMPC(model=mpc_model, horizon=N, dt=dt)
    x = np.zeros(3)
    u = np.zeros(2)
    x_d = np.array([10, 10, 0])

    for _ in range(10):
        x_list, u_list = controller.step(x, u, x_d)
        x, u = x_list[:, -1], u_list[:, -1]
        print(f"x_list: {x_list}")
        print(f"u_list: {u_list}")
        print(f"x: {x}")
        print(f"u: {u}")
