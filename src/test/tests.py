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
    N = 40  # Time horizon

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


def dubins_distance_direct_collocation_example():
    """
    Direct collocation method 

    Based on the work of Joel Andersson, Joris Gillis and Moriz Diehl at KU Leuven

    Links:
    https://github.com/casadi/casadi/blob/main/docs/examples/matlab/direct_collocation_opti.m
    and
    https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py


    """
    model = DubinsCarModel

    # Degree of interpolating polynomial
    d = 3

    # Get collocation points
    tau_root = np.append(0, ca.collocation_points(d, "legendre"))

    # Collocation, continuity abd quadrature coefficients
    C, D, B = np.zeros((d+1, d+1)), np.zeros(d+1), np.zeros(d+1)

    for j in range(d+1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / \
                    (tau_root[j]-tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d+1):
            C[j, r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    # Setup states
    x_pos = ca.SX.sym("x")
    y_pos = ca.SX.sym("y")
    theta = ca.SX.sym("theta")
    x = ca.vertcat(x_pos,
                   y_pos,
                   theta)

    # Setup inputs
    v = ca.SX.sym("v")
    phi = ca.SX.sym("phi")
    u = ca.vertcat(v,
                   phi)

    # Model
    xdot = model.step(x, u)

    # Objective function
    L = x_pos**2 + y_pos**2 + theta**2 + v**2 + phi**2

    # Continuous time dynamics
    f = ca.Function("f", [x, u], [xdot, L], ["x", "u"], ["xdot", "L"])

    # Discretization
    N = 40      # Control intervals
    dt = 0.05   # Time step length

    # Initialize empty NLP
    opti = Optimizer()
    J = 0

    # Don't know what it means but "lift" initial conditions
    Xk = opti.variable(3)
    opti.subject_to(Xk == ca.vertcat(0, 0, 0))
    opti.set_initial(Xk == ca.vertcat(0, 0, 0))

    # Apparently collect all states/controls
    Xs = [Xk]
    Us = []

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for control
        Uk = opti.variable(2)
        Us.append(Uk)
        opti.subject_to(opti.bounded(-1, Uk[0], 1))
        opti.subject_to(opti.bounded(D2R(-15), Uk[1], D2R(15)))
        opti.set_initial(Uk[0], 0)
        opti.set_initial(Uk[1], 0)

        # Decision variables for helper states at each collocation point
        Xc = opti.variable(3, d)
        # Don't know if this constraint is needed
        opti.subject_to(Xc, np.tile([-np.inf, np.inf], d))
        opti.set_initial(Xc, np.zeros((3, d)))

        # Evaluate ODE right-hand-side at all helper states
        ode, quad = f(Xc, Uk)

        # Add contribution to quadrature function
        J += quad*B*dt

        # Get interpolating points of collocation polynomial

    # TODO: Finnish this python/direct collocation/opti example


def test_mpc():
    dt = 0.05
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
    control_fps = 20
    sim_fps = 60
    N = 40
    eta_init = np.array([5, 0, 0, 0, 0, 0],
                        float)           # 3 DOF example
    eta_d = np.array([25/2-0.75-1, 0, 0, 0, 0, 0], float)
    mpc_config = {"N": N,
                  "dt": 1/control_fps,
                  "Q": np.diag([100, 100, 100]),
                  "Q_slack": np.diag([1, 1, 1]),
                  "R": np.diag([1, 1])}

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
