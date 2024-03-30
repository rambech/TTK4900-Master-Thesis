import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from control import NMPC, Manual
from control.optimizer import Optimizer
from vehicle.models import DubinsCarModel, OtterModel
from utils import D2R
from plotting import plot_solution
from vehicle import DubinsCar, Otter
from maps import SimpleMap, Target
from simulator import Simulator


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
    N = 400  # Time horizon

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
    opti.simple_quadratic(x, x_d)
    # opti.minimize(ca.sum2(x[0, 1:N+1]-x_d[0, 1:N+1])**2 +
    #               ca.sum2(x[1, 1:N+1]-x_d[1, 1:N+1])**2 +
    #               ca.sum2(x[2, 1:N+1]-x_d[2, 1:N+1])**2)

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
    opti.subject_to(v[0] == 0)
    opti.subject_to(phi[0] == 0)
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
    N = 400  # Time horizon

    # Making optimization object
    opti = Optimizer()

    u_init = np.zeros(2)
    x_init = np.zeros(3)
    x_desired = np.tile([10, 10, 0], (N+1, 1)).tolist()
    x_d = ca.hcat(x_desired)

    mpc_model = DubinsCarModel(dt=dt, N=N)
    x, u, s = mpc_model.single_shooting(x_init, u_init, opti)

    # Objective
    # opti.quadratic(x, u, x_d)
    opti.simple_quadratic(x, x_d)

    # Setup solver and solve
    opti.solver('ipopt')
    solution = opti.solve()

    print(f"x: {solution.value(x)}")

    print(f"x_pos_opt {solution.value(x[0])}")
    print(f"y_pos_opt {solution.value(x[1])}")

    plot_solution(solution, x, u)


def otter_distance_example():
    mpc_model = OtterModel()
    N = 100  # Time horizon

    # Making optimization object
    opti = Optimizer()

    # Declaring optimization variables
    # State variables
    x = opti.variable(6, N+1)
    north = x[0, :]
    east = x[1, :]
    yaw = x[2, :]
    surge = x[3, :]
    sway = x[4, :]
    yaw_rate = x[5, :]

    # Input variables
    u = opti.variable(2, N)
    port_u = u[0, :]
    starboard_u = u[1, :]

    # Slack variables
    s = opti.variable(3, N+1)

    x_desired = np.tile([10, 10, 0], (N+1, 1)).tolist()
    x_d = ca.hcat(x_desired)

    # Objective
    opti.simple_quadratic(x, x_d)

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

        if k > 0:
            opti.subject_to(opti.bounded(-100*dt,
                                         port_u[k] - port_u[k-1],
                                         100*dt))
            opti.subject_to(opti.bounded(-100*dt,
                                         starboard_u[k] - starboard_u[k-1],
                                         100*dt))

    # NOTE: As of 30th of March using thruster dynamics made the
    #       solution trajectories worse
    # kx_1, ku_1 = mpc_model.step(x[:, 0],               u[:, 0], [0, 0])
    # kx_2, ku_2 = mpc_model.step(x[:, 0] + dt/2 * kx_1, u[:, 0], [0, 0])
    # kx_3, ku_3 = mpc_model.step(x[:, 0] + dt/2 * kx_2, u[:, 0], [0, 0])
    # kx_4, ku_4 = mpc_model.step(x[:, 0] + dt * kx_3,   u[:, 0], [0, 0])
    # x_next = x[:, 0] + dt/6 * (kx_1+2*kx_2+2*kx_3+kx_4)
    # opti.subject_to(x[:, 1] == x_next)

    # for k in range(1, N):
    #     kx_1, ku_1 = mpc_model.step(x[:, k],               u[:, k], u[:, k-1])
    #     kx_2, ku_2 = mpc_model.step(x[:, k] + dt/2 * kx_1, u[:, k], u[:, k-1])
    #     kx_3, ku_3 = mpc_model.step(x[:, k] + dt/2 * kx_2, u[:, k], u[:, k-1])
    #     kx_4, ku_4 = mpc_model.step(x[:, k] + dt * kx_3,   u[:, k], u[:, k-1])
    #     x_next = x[:, k] + dt/6 * (kx_1+2*kx_2+2*kx_3+kx_4)
    #     opti.subject_to(x[:, k+1] == x_next)

    #     if k != N-1:
    #         u_next = u[:, k] + dt/6 * (ku_1+2*ku_2+2*ku_3+ku_4)
    #         opti.subject_to(u[:, k+1] == u_next)

    # Control signal and time constraint
    opti.subject_to(opti.bounded(-100, port_u, 100))
    opti.subject_to(opti.bounded(-100, starboard_u, 100))

    # Boundary values
    # Initial conditions
    opti.subject_to(north[0] == 0.0)
    opti.subject_to(east[0] == 0.0)
    opti.subject_to(yaw[0] == 0.0)
    opti.subject_to(surge[0] == 0.0)
    opti.subject_to(sway[0] == 0.0)
    opti.subject_to(yaw_rate[0] == 0.0)
    opti.subject_to(port_u[0] == 0.0)
    opti.subject_to(starboard_u[0] == 0.0)

    opti.set_initial(north, 0.0)
    opti.set_initial(east, 0.0)
    opti.set_initial(yaw, 0.0)
    opti.set_initial(port_u, 0.0)
    opti.set_initial(starboard_u, 0.0)

    # Setup solver and solve
    opti.solver('ipopt')
    solution = opti.solve()

    print(f"x_pos_opt {solution.value(north)}")
    print(f"y_pos_opt {solution.value(east)}")
    # print(f"yaw_opt {len(solution.value(opti.g))}")

    plot_solution(solution, x, u)


def test_mpc():
    dt = 0.2
    N = 40
    config = {"N": N,
              "dt": dt,
              "Q": np.diag([1, 1, 1]),
              "R": np.diag([1, 1])}
    mpc_model = DubinsCarModel(N=N, dt=dt)
    controller = NMPC(model=mpc_model, config=config)
    x = np.zeros(3)
    u = np.zeros(2)
    x_d = np.array([10, 10, 0])

    for _ in range(3):
        x_list, u_list = controller.step(x, u, x_d)
        x, u = x_list[:, -1], u_list[:, -1]
        print(f"x_list: {x_list}")
        print(f"u_list: {u_list}")
        print(f"x: {x}")
        print(f"u: {u}")


def test_mpc_simulator():
    """
    Procedure for testing simulator
    """

    # TODO: Put in a parser argument "Press enter to start"

    # Initialize constants
    control_fps = 10
    sim_fps = 30
    N = 40
    eta_init = np.array([-5, 0, 0, 0, 0, 0],
                        float)           # 3 DOF example
    eta_d = np.array([25/2-0.75-1, 0, 0, 0, 0, 0], float)
    mpc_config = {"N": N,
                  "dt": 1/control_fps,
                  "Q": np.diag([100, 100, 100]).tolist(),
                  "Q_slack": np.diag([1, 1, 1]).tolist(),
                  "R": np.diag([1, 1]).tolist()}

    # Initialize vehicle and control
    vehicle = Otter(dt=1/sim_fps)
    model = OtterModel(dt=1/control_fps, N=N)
    controller = NMPC(model=model, config=mpc_config)

    # Initialize map and objective
    map = SimpleMap()
    target = Target(eta_d, vehicle, map.origin)

    # Simulate
    simulator = Simulator(vehicle, controller, map, None, target,
                          eta_init=eta_init, fps=control_fps, data_acq=True)
    simulator.simulate()


def test_simulator():
    """
    Procedure for testing simulator
    """
    # Initialize constants
    control_fps = 20
    sim_fps = 60
    N = 40
    eta_init = np.array([0, 0, 0, 0, 0, 0], float)           # 3 DOF example
    eta_d = np.array([25/2-0.75-1, 0, 0, 0, 0, 0], float)
    mpc_config = {"N": N,
                  "dt": 1/control_fps,
                  "Q": np.diag([1, 1, 1]),
                  "R": np.diag([1, 1])}

    # Initialize vehicle and control
    vehicle = Otter(dt=1/sim_fps)
    model = DubinsCarModel(dt=1/control_fps, N=N)
    controller = Manual()  # NMPC(model=model, config=mpc_config)

    # Initialize map and objective
    map = SimpleMap()
    target = Target(eta_d, vehicle, map.origin)

    # Simulate
    simulator = Simulator(vehicle, controller, map, None, target,
                          eta_init=eta_init, fps=control_fps)
    simulator.simulate()


def test_v2c():
    from utils import V2C

    harbour = np.array([[11.75, -2.5],
                        [11.75, 2.5],
                        [9, 15],
                        [-12.5, 15],
                        [-12.5, -15],
                        [9, -15]])

    A, b = V2C(harbour)
    print(f"A:\n {np.round(A, 4)}")
    print(f"b:\n {np.round(b, 4)}")
