import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from control import NMPC, Manual, RLNMPC
from control.optimizer import Optimizer
from vehicle.models import DubinsCarModel, OtterModel
from utils import D2R
import utils
from plotting import plot_solution
from vehicle import DubinsCar, Otter, SimpleOtter
from maps import Brattora, Ravnkloa, Nidelva, SimpleMap, Target
from simulator import Simulator
import time


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
    N = 50  # Time horizon
    dt = 0.2    # Time step

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

    # x_desired = np.tile([10, 0, 0], (N+1, 1)).tolist()
    x_desired = [10, 10, 0]
    x_d = ca.hcat(x_desired)

    # Objective
    opti.simple_quadratic(x, x_d)
    # opti.minimize(ca.sum2(x[0, 1:N+1]-x_d[0, 1:N+1])**2 +
    #               ca.sum2(x[1, 1:N+1]-x_d[1, 1:N+1])**2 +
    #               ca.sum2(x[2, 1:N+1]-x_d[2, 1:N+1])**2)

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

    plot_solution(dt, solution, x, u)


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

    plot_solution(dt, solution, x, u)


def otter_distance_example():
    mpc_model = OtterModel()
    N = 30  # Step horizon
    dt = 0.5  # Time step

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

    # x_desired = np.tile([[10], [0], [0]], (1, N+1)).tolist()
    # print(f"x_desired: {x_desired}")
    # print(f"x_desired.shape: {x_desired.shape}")
    x_desired = [10, 10, 0]
    x_d = ca.hcat(x_desired)

    # Objective
    opti.simple_quadratic(x, x_d)

    # Fixed step Runge-Kutta 4 integrator
    for k in range(N):
        kx_1 = mpc_model.step(x[:, k],               u[:, k])
        kx_2 = mpc_model.step(x[:, k] + dt/2 * kx_1, u[:, k])
        kx_3 = mpc_model.step(x[:, k] + dt/2 * kx_2, u[:, k])
        kx_4 = mpc_model.step(x[:, k] + dt * kx_3,   u[:, k])
        x_next = x[:, k] + dt/6 * (kx_1+2*kx_2+2*kx_3+kx_4)
        opti.subject_to(x[:, k+1] == x_next)

    # Control signal and time constraint
    opti.subject_to(opti.bounded(-70, port_u, 100))
    opti.subject_to(opti.bounded(-70, starboard_u, 100))
    opti.subject_to(opti.bounded(-dt*100,
                                 port_u[:, 1:] - port_u[:, :-1],
                                 dt*100))
    opti.subject_to(opti.bounded(-dt*100,
                                 starboard_u[:, 1:] - starboard_u[:, :-1],
                                 dt*100))

    # Boundary values
    # Initial conditions
    # TODO: Something is defined wrong,
    # because different initial conditions on yaw
    # give wack results
    opti.subject_to(north[0] == 0.001)
    opti.subject_to(east[0] == 0.001)
    opti.subject_to(yaw[0] == D2R(0.0001))
    opti.subject_to(surge[0] == 0.001)
    opti.subject_to(sway[0] == 0.001)
    opti.subject_to(yaw_rate[0] == 0.001)
    opti.subject_to(port_u[0] == 0.001)
    opti.subject_to(starboard_u[0] == 0.001)

    opti.set_initial(north, 0.0)
    opti.set_initial(east, 0.0)
    opti.set_initial(yaw, 0.0)
    opti.set_initial(port_u, 0.0)
    opti.set_initial(starboard_u, 0.0)

    # Setup solver and solve
    opts = {'ipopt.print_level': 5, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.warm_start_init_point': 'yes'}
    opti.solver('ipopt', opts)
    solution = opti.solve()

    print(f"x_pos_opt {solution.value(north)}")
    print(f"y_pos_opt {solution.value(east)}")
    # print(f"yaw_opt {len(solution.value(opti.g))}")

    plot_solution(dt, solution, x, u)


def tol_reached(x, x_d, pos_tol, head_tol) -> bool:
    if (-pos_tol <= np.linalg.norm(x[:2] - x_d[:2], 2) <= pos_tol
            and -head_tol <= utils.ssa(x[2] - x_d[2]) <= head_tol):

        return True

    return False


def test_mpc():
    control_fps = 5  # 2.5
    dt = 1/control_fps
    N = 50

    conventional = True
    rl = False
    num_steps = 70
    plot_bool = True
    acq = True

    # harbour_geometry = [[-.25,  -.25],
    #                     [10.25,  -.25],
    #                     [10.25, 10.25],
    #                     [-.25, 10.25]]
    # harbour_geometry = [[10, -15],
    #                     [11.75, -5],
    #                     [11.75, 5],
    #                     [10, 15],
    #                     [-12.5, 15],
    #                     [-12.5, -15]]
    harbour_geometry = [[15, -42.5],
                        [40, -12.5],
                        [-7.5, 30],
                        [-16.5, 25.5],
                        [-26, 15],
                        [-30, -2]]
    harbour_space = utils.V2C(harbour_geometry)
    # config = {
    #     "N": N,
    #     "dt": 0.2,
    #     "Q": np.diag([1, 10, 50]).tolist(),
    #     "q_slack": [100, 100, 100, 100, 100, 100],
    #     "R": np.diag([0.01, 0.01]).tolist(),
    #     "delta": 10,
    #     "q_xy": 20,
    #     "q_psi": 100,
    #     "gamma": 0.95,
    #     "alpha": 0.01,  # RL Learning rate
    #     "beta": 0.01  # SYSID Learning rate
    # }

    # config = {
    #     "N": N,
    #     "dt": 0.2,
    #     "Q": np.diag([1, 10, 20]).tolist(),
    #     "q_slack": [100, 100, 100, 100, 100, 100],
    #     "R": np.diag([0.01, 0.01]).tolist(),
    #     "delta": 10,
    #     "q_xy": 20,
    #     "q_psi": 100,
    #     "gamma": 0.95,
    #     "alpha": 0.01,  # RL Learning rate
    #     "beta": 0.01  # SYSID Learning rate
    # }

    # config = {
    #     "N": N,
    #     "dt": 1/control_fps,
    #     "Q": np.diag([100, 1, 1]).tolist(),
    #     "q_slack": [100, 100, 10, 10, 10, 10, 10],
    #     "R": np.diag([0.05, 0.05]).tolist(),
    #     "delta": 5,
    #     "q_xy": 20,
    #     "q_psi": 100,
    #     "gamma": 0.95,
    #     "alpha": 0.01,  # RL Learning rate
    #     "beta": 0.01,  # SYSID Learning rate
    #     "batch_size": 10
    # }

    config = {
        "N": N,
        "dt": 1/control_fps,
        "Q": np.diag([0, 0, 0]).tolist(),
        "q_slack": [100, 100, 100, 100, 100, 100, 1000],
        "R": np.diag([0.01, 0.01]).tolist(),
        "delta": 1,
        "q_xy": 30,
        "q_psi": 20,
        "alpha": 0,  # 0.01,
        "beta": 0.05,
        "gamma": 0.95,
        "batch size": 0,  # 10,
        "lq": 0.1,  # Make Q-hessian estimate positive definite
        "lf": 0.1   # Make PEM hessian estimate positive definite
    }

    data = {
        "Config": config,
    }

    if conventional:
        mpc_model = OtterModel(N=N, dt=dt)
        controller = NMPC(model=mpc_model, config=config,
                          space=harbour_space, use_slack=False)
    if rl:
        rlnmpc_model = OtterModel(N=N, dt=dt)
        rl_controller = RLNMPC(model=rlnmpc_model, config=config,
                               space=harbour_space, use_slack=False)
    # controller = NMPC(model=mpc_model, config=config)
    # print(f"A: {harbour_space[0]}")
    # print(f"b: {harbour_space[1]}")
    # x = 0.001*np.ones(6)
    u = 0.001*np.ones(2)
    # x = np.array([-5, 5, 0.001, 0.001, 0.001, 0.001])
    x = np.array([23.240456, -20.00666667, utils.D2R(137.37324840062468),
                 0.001, 0.001, 0.001])
    u_rl = u.copy()
    x_rl = x.copy()
    # x_d = np.array([25/2-0.75-0.25, 0, -np.pi/2])
    x_d = np.array([-20.36019, 19.44486, utils.D2R(137.37324840062468)])
    # x_d = np.array([25/2-0.75-0.5, 0, 0])

    time_list = []
    rl_time_list = []
    data["Path"] = [x[:3].tolist()]
    data["u"] = [u.tolist()]
    data["state predictions"] = []
    data["control predictions"] = []
    pos_tol = .5
    head_tol = utils.D2R(15)
    print(f"Position tolarance: {pos_tol}")
    print(f"Heading tolerance: {head_tol}")

    for _ in range(num_steps):
        # Conventional NMPC test
        if conventional:
            t0 = time.time()
            try:
                x_list, u_list, _, _ = controller.debug(x, u, x_d)
            except RuntimeError as error:
                print("Error caught", error)

                from plotting import plot
                if conventional:
                    plot(dt, x_list, u_list)

                break

            t1 = time.time()

            t = t1 - t0
            time_list.append(t)

            data["state predictions"].append(x_list.tolist())
            data["control predictions"].append(u_list.tolist())

            x, u = x_list[:, 1], u_list[:, 1]
            x_N, u_N = x_list[:, -1], u_list[:, -1]
            data["Path"].append(x[:3].tolist())
            data["u"].append(u.tolist())

        # RL NMPC test
        if rl:
            rl_t0 = time.time()
            rl_x_list, rl_u_list = rl_controller.debug(x_rl, u_rl, x_d)
            rl_t1 = time.time()

            rl_t = rl_t1 - rl_t0
            rl_time_list.append(rl_t)

            x_rl, u_rl = rl_x_list[:, 1], rl_u_list[:, 1]

        if plot_bool and tol_reached(x, x_d, pos_tol, head_tol):
            from plotting import plot
            if conventional:
                plot(dt, x_list, u_list)

            if rl:
                plot(dt, rl_x_list, rl_u_list)

            break

        # print(f"x_list: {x_list}")
        # print(f"u_list: {u_list}")
        if conventional:
            print(f"== Conventional NMPC ==")
            print(f"x: {np.round(x, 5)}")
            print(f"u: {np.round(u, 5)}")
            print(f"distance error: {np.linalg.norm(x[:2]-x_d[:2], 2)}")
            print(f"heading error: {utils.ssa(x[2]-x_d[2])}")
            print(f"t: {np.round(t, 5)}")

        if rl:
            print(f"======= RL NMPC =======")
            print(f"x: {np.round(x_rl, 5)}")
            print(f"u: {np.round(u_rl, 5)}")
            print(f"distance error: {np.linalg.norm(x_rl[:2]-x_d[:2], 2)}")
            print(f"heading error: {utils.ssa(x_rl[2]-x_d[2])}")
            print(f"t: {np.round(rl_t, 5)}")

    print("=======================")
    print("== End state reached ==")
    print("=======================")

    if conventional:
        print("== Conventional NMPC ==")
        print(f"Average time used: {np.mean(time_list)}")
        print(f"Std time used: {np.std(time_list)}")
        print(f"Max time used: {np.max(time_list)}")
        print(f"Min time used: {np.min(time_list)}")

    if rl:
        print("======= RL NMPC =======")
        print(f"Average time used: {np.mean(rl_time_list)}")
        print(f"Std time used: {np.std(rl_time_list)}")
        print(f"Max time used: {np.max(rl_time_list)}")
        print(f"Min time used: {np.min(rl_time_list)}")

    # ======================
    # == Data acquisition ==
    # ======================
    if acq:
        from plotting.data import save_data

        save_data(data, "test_mpc")


def test_mpc_simulator():
    """
    Procedure for testing simulator
    """

    # TODO: Put in a parser argument "Press enter to start"

    # Initialize constants
    control_fps = 2.5
    sim_fps = 50
    N = 50
    scale = 30
    # eta_init = np.array([-5, 5, 0.0001, 0.0001, 0.0001, 0.0001],
    #                     float)           # 3 DOF example

    eta_init = np.array([0, 0, 0.0001, 0.0001, 0.0001, np.pi],
                        float)           # 3 DOF example

    # Forward docking goal
    # eta_d = np.array([25/2-0.75-1, 0, 0], float)

    # Backward docking goal
    # eta_d = np.array([25/2-0.75-0.5, 0, np.pi], float)

    # Sideways docking goal
    # eta_d = np.array([25/2-0.75-0.5, 0, -np.pi/2], float)

    eta_d = np.array([-10, -10, -np.pi/2])

    harbour_geometry = [[10, -15],
                        [11.75, -5],
                        [11.75, 5],
                        [10, 15],
                        [-12.5, 15],
                        [-12.5, -15]]

    harbour_space = utils.V2C(harbour_geometry)

    # mpc_config = {
    #     "N": N,
    #     "dt": 1/control_fps,
    #     "Q": np.diag([1, 10, 50]).tolist(),
    #     "q_slack": [100, 100, 100, 100, 100, 100],
    #     "R": np.diag([0.01, 0.01]).tolist(),
    #     "delta": 5,
    #     "q_xy": 20,
    #     "q_psi": 100
    # }

    mpc_config = {
        "N": N,
        "dt": 1/control_fps,
        "Q": np.diag([100, 1, 1]).tolist(),
        "q_slack": [100, 100, 10, 10, 10, 10, 10],
        "R": np.diag([0.05, 0.05]).tolist(),
        "delta": 5,
        "q_xy": 20,
        "q_psi": 200
    }

    # mpc_config = {
    #     "N": N,
    #     "dt": 1/control_fps,
    #     "Q": np.diag([10, 0.1, 1]).tolist(),
    #     "q_slack": [1, 1, 1, 1, 1, 1],
    #     "R": np.diag([0.05, 0.05]).tolist(),
    #     "delta": 10,
    #     "q_xy": 10,
    #     "q_psi": 10
    # }

    # Initialize vehicle and control
    vehicle = Otter(dt=1/sim_fps)
    model = OtterModel(dt=1/control_fps, N=N)
    controller = NMPC(model=model, config=mpc_config,
                      space=harbour_space, use_slack=False)

    # Initialize map and objective
    # map = SimpleMap(harbour_geometry)
    map = Brattora(harbour_geometry)
    target = Target(eta_d, vehicle, map)

    # Simulate
    simulator = Simulator(vehicle, controller, map, None, target,
                          eta_init=eta_init, fps=control_fps,
                          data_acq=True, render=True)
    simulator.simulate()


def test_brattora(rl=True, default=True, estimate_current=True, V_c=0, simple=False, B=1, sysid=True):
    """
    A simpler vehicle model
    that is an exactly the same as the NMPC model

    """

    print("===================================")
    print("-------- Running Brattøra ---------")

    # Initialize constants
    control_fps = 5
    sim_fps = 50
    N = 50
    beta_c = 0
    speed_limit = 5

    if sysid:
        beta = 0.005
    else:
        beta = 0

    # Initial pose
    eta_init = np.array([23.240456, -20.00666667, 0, 0,
                        0, utils.D2R(137.37324840062468)])

    # Forward docking goal
    eta_d = np.array([-20.36019, 19.44486, utils.D2R(137.37324840062468)])

    print(f"initial heading in test: {eta_init[-1]}")
    print(f"desired heading in test: {eta_d[-1]}")

    harbour_geometry = [[15, -42.5],
                        [40, -12.5],
                        [-7.5, 30],
                        [-16.5, 25.5],
                        [-26, 15],
                        [-30, -2]]
    harbour_space = utils.V2C(harbour_geometry)

    # TODO: Make a feature for .ini, .json or .yaml config file

    nmpc_config = {
        "Name": "NMPC config",
        "N": N,
        "dt": 1/control_fps,
        "V_c": V_c,
        "beta_c": beta_c,
        "Q": np.diag([1, 100, 10]).tolist(),
        "q_slack": [10000, 10000, 100, 100, 100, 100, 1000],
        "R": np.diag([0.04, 0.04]).tolist(),
        "delta": 1,
        "q_xy": 100,
        "q_psi": 150,
        "alpha": 0,
        "beta": 0,
        "gamma": 1,
        "projection threshold": 0.01,
        "speed limit": speed_limit
    }

    # TODO: Make a feature for .ini, .json or .yaml config file

    rlnmpc_config = {
        "Name": "RL config",
        "N": N,
        "dt": 1/control_fps,
        "V_c": V_c,
        "beta_c": beta_c,
        "Q": np.diag([1, 100, 10]).tolist(),
        "q_slack": [1000, 1000, 100, 100, 100, 100, 1000],
        "R": np.diag([0.04, 0.04]).tolist(),
        "delta": 1,
        "q_xy": 100,
        "q_psi": 150,
        "alpha": 0.005,
        "beta": beta,
        "gamma": 0.99,
        "batch size": B,
        "lq": 0.1,  # Make Q-hessian estimate positive definite
        "lf": 0.1,   # Make PEM hessian estimate positive definite
        "projection threshold": 0.01,
        "speed limit": speed_limit
    }

    # Initialize vehicle and control
    if simple:
        vehicle = SimpleOtter(dt=1/sim_fps)
    else:
        vehicle = Otter(dt=1/sim_fps)

    # Initialize map and objective
    quay_edge = harbour_geometry[3], harbour_geometry[4]
    map = Brattora(harbour_geometry, quay_edge, V_c, beta_c)
    target = Target(eta_d, vehicle, map)

    if rl:
        print("----------- RL-NMPC Test -----------")
        model = OtterModel(dt=1/control_fps, N=N, buffer=0.2,
                           default=default, estimate_current=estimate_current)
        controller = RLNMPC(model=model, config=rlnmpc_config, type="tracking",
                            space=harbour_space, use_slack=False)
        rlnmpc_config["actual theta"] = vehicle.theta.tolist()
        rlnmpc_config["initial theta"] = model.theta.tolist()
        rlnmpc_config["vehicle type"] = type(vehicle).__name__
        print("Config:")
        print(rlnmpc_config)
    else:
        print("------------ NMPC Test ------------")
        model = OtterModel(dt=1/control_fps, N=N, buffer=0.2,
                           default=True, estimate_current=False)
        controller = RLNMPC(model=model, config=nmpc_config, type="setpoint",
                            space=harbour_space, use_slack=False)
        nmpc_config["actual theta"] = vehicle.theta.tolist()
        nmpc_config["initial theta"] = model.theta.tolist()
        nmpc_config["vehicle type"] = type(vehicle).__name__
        print("Config:")
        print(nmpc_config)

    simulator = Simulator(vehicle, controller, map, planner=None,
                          target=target, eta_init=eta_init, fps=control_fps,
                          data_acq=True, render=True)

    # Simulate
    simulator.simulate()

    print("-------- Stopping Brattøra --------")
    print("===================================")


def test_ravnkloa(rl=True, default=True, estimate_current=True, V_c=0, simple=False, B=1, sysid=True):
    """
    Test using map of Ravnkloa

    """

    print("===================================")
    print("-------- Running Ravnkloa ---------")

 # Initialize constants
    control_fps = 5
    sim_fps = 50
    N = 50
    speed_limit = 5  # [kts]
    beta_c = utils.D2R(-130)

    if sysid:
        beta = 0.005
    else:
        beta = 0

    # Initial pose
    # eta_init = np.array([-15, -30, 0, 0,
    #                     0, utils.D2R(50)])

    # eta_init = np.array([-5, -10, 0, 0,
    #                     0, utils.D2R(50)])
    eta_init = np.array([-7, -15, 0, 0,
                        0, utils.D2R(50)])

    # Forward docking goal
    eta_d = np.array([-11, 36.5, utils.D2R(165)])
    # eta_d = np.array([-5, -15, utils.D2R(50)])

    print(f"initial heading in test: {eta_init[-1]}")
    print(f"desired heading in test: {eta_d[-1]}")

    harbour_geometry = [[-10, 47],
                        [-14, 30],
                        [-20, -45],
                        [37, 40]]
    harbour_space = utils.V2C(harbour_geometry)

    # TODO: Make a feature for .ini or .yaml config file

    nmpc_config = {
        "Name": "NMPC config",
        "N": N,
        "dt": 1/control_fps,
        "V_c": V_c,
        "beta_c": beta_c,
        "Q": np.diag([1, 100, 10]).tolist(),
        "q_slack": [10000, 10000, 100, 100, 100, 100, 1000],
        "R": np.diag([0.04, 0.04]).tolist(),
        "delta": 1,
        "q_xy": 100,
        "q_psi": 150,
        "alpha": 0,
        "beta": 0,
        "gamma": 1,
        "projection threshold": 0.01,
        "speed limit": speed_limit
    }

    # TODO: Make a feature for .ini, .json or .yaml config file

    rlnmpc_config = {
        "Name": "RL config",
        "N": N,
        "dt": 1/control_fps,
        "V_c": V_c,
        "beta_c": beta_c,
        "Q": np.diag([1, 100, 10]).tolist(),
        "q_slack": [1000, 1000, 100, 100, 100, 100, 1000],
        "R": np.diag([0.04, 0.04]).tolist(),
        "delta": 1,
        "q_xy": 100,
        "q_psi": 150,
        "alpha": 0.005,
        "beta": beta,
        "gamma": 0.99,
        "batch size": B,
        "lq": 0.1,  # Make Q-hessian estimate positive definite
        "lf": 0.1,   # Make PEM hessian estimate positive definite
        "projection threshold": 0.01,
        "speed limit": speed_limit
    }

    # Initialize vehicle and control
    if simple:
        vehicle = SimpleOtter(dt=1/sim_fps)
    else:
        vehicle = Otter(dt=1/sim_fps)

    # Initialize map and objective
    quay_edge = harbour_geometry[0], harbour_geometry[1]
    map = Ravnkloa(harbour_geometry, quay_edge, V_c, beta_c)
    target = Target(eta_d, vehicle, map)

    if rl:
        print("----------- RL-NMPC Test -----------")
        model = OtterModel(dt=1/control_fps, N=N, buffer=0.2,
                           default=default, estimate_current=estimate_current)
        controller = RLNMPC(model=model, config=rlnmpc_config, type="tracking",
                            space=harbour_space, use_slack=False)
        rlnmpc_config["actual theta"] = vehicle.theta.tolist()
        rlnmpc_config["initial theta"] = model.theta.tolist()
        rlnmpc_config["vehicle type"] = type(vehicle).__name__
        print("Config:")
        print(rlnmpc_config)
    else:
        print("------------ NMPC Test ------------")
        model = OtterModel(dt=1/control_fps, N=N, buffer=0.2,
                           default=True, estimate_current=False)
        controller = RLNMPC(model=model, config=nmpc_config, type="setpoint",
                            space=harbour_space, use_slack=False)
        nmpc_config["actual theta"] = vehicle.theta.tolist()
        nmpc_config["initial theta"] = model.theta.tolist()
        nmpc_config["vehicle type"] = type(vehicle).__name__
        print("Config:")
        print(nmpc_config)

    simulator = Simulator(vehicle, controller, map, planner=None,
                          target=target, eta_init=eta_init, fps=control_fps,
                          data_acq=True, render=True)

    # Simulate
    simulator.simulate()

    print("-------- Stopping Ravnkloa --------")
    print("===================================")


def test_nidelva(rl=True, default=True, estimate_current=True, V_c=0, simple=False, B=1, sysid=True):
    """
    A simpler vehicle model
    that is an exactly the same as the NMPC model

    """

    print("===================================")
    print("--------- Running Nidelva ---------")

    # Initialize constants
    control_fps = 5
    sim_fps = 50
    N = 50
    speed_limit = 5  # [kts]
    beta_c = utils.D2R(10)

    if sysid:
        beta = 0.005
    else:
        beta = 0

    # Initial pose
    eta_init = np.array([-25, -20, 0, 0,
                        0, 0])

    # Forward docking goal
    eta_d = np.array(
        [25.406768959337686, 31.143935018607795, 0.10626486289107881])
    # eta_d = np.array([26.6, 31.25, 0.10626486289107881])

    print(f"initial heading in test: {eta_init[-1]}")
    print(f"desired heading in test: {eta_d[-1]}")

    harbour_geometry = [[20, -30],
                        [27, 27.5],
                        [26.2, 35],
                        [-30, 35],
                        [-30, -30]]
    harbour_space = utils.V2C(harbour_geometry)

    # TODO: Find a good tuning for NMPC

    nmpc_config = {
        "Name": "NMPC config",
        "N": N,
        "dt": 1/control_fps,
        "V_c": V_c,
        "beta_c": beta_c,
        "Q": np.diag([1, 100, 10]).tolist(),
        "q_slack": [10000, 10000, 100, 100, 100, 100, 1000],
        "R": np.diag([0.04, 0.04]).tolist(),
        "delta": 1,
        "q_xy": 100,
        "q_psi": 150,
        "alpha": 0,
        "beta": 0,
        "gamma": 1,
        "projection threshold": 0.01,
        "speed limit": speed_limit
    }

    # TODO: Make a feature for .ini, .json or .yaml config file

    rlnmpc_config = {
        "Name": "RL config",
        "N": N,
        "dt": 1/control_fps,
        "V_c": V_c,
        "beta_c": beta_c,
        "Q": np.diag([1, 100, 10]).tolist(),
        "q_slack": [1000, 1000, 100, 100, 100, 100, 1000],
        "R": np.diag([0.04, 0.04]).tolist(),
        "delta": 1,
        "q_xy": 100,
        "q_psi": 150,
        "alpha": 0.005,
        "beta": beta,
        "gamma": 0.99,
        "batch size": B,
        "lq": 0.1,  # Make Q-hessian estimate positive definite
        "lf": 0.1,   # Make PEM hessian estimate positive definite
        "projection threshold": 0.01,
        "speed limit": speed_limit
    }

    # Initialize vehicle and control
    if simple:
        vehicle = SimpleOtter(dt=1/sim_fps)
    else:
        vehicle = Otter(dt=1/sim_fps)

    # Initialize map and objective
    quay_edge = harbour_geometry[1], harbour_geometry[2]
    map = Nidelva(harbour_geometry, quay_edge, V_c, beta_c)
    target = Target(eta_d, vehicle, map)

    if rl:
        print("----------- RL-NMPC Test -----------")
        model = OtterModel(dt=1/control_fps, N=N, buffer=0.2,
                           default=default, estimate_current=estimate_current)
        controller = RLNMPC(model=model, config=rlnmpc_config, type="tracking",
                            space=harbour_space, use_slack=False)
        rlnmpc_config["actual theta"] = vehicle.theta.tolist()
        rlnmpc_config["initial theta"] = model.theta.tolist()
        rlnmpc_config["vehicle type"] = type(vehicle).__name__
        print("Config:")
        print(rlnmpc_config)
    else:
        print("------------ NMPC Test ------------")
        model = OtterModel(dt=1/control_fps, N=N, buffer=0.2,
                           default=True, estimate_current=False)
        controller = RLNMPC(model=model, config=nmpc_config, type="setpoint",
                            space=harbour_space, use_slack=False)
        nmpc_config["actual theta"] = vehicle.theta.tolist()
        nmpc_config["initial theta"] = model.theta.tolist()
        nmpc_config["vehicle type"] = type(vehicle).__name__
        print("Config:")
        print(nmpc_config)

    simulator = Simulator(vehicle, controller, map, planner=None,
                          target=target, eta_init=eta_init, fps=control_fps,
                          data_acq=True, render=True)

    # Simulate
    simulator.simulate()

    print("--------- Stopping Nidelva --------")
    print("===================================")


def test_simulator():
    """
    Procedure for testing simulator
    """
    # Initialize constants
    control_fps = 5
    sim_fps = 60
    N = 40
    eta_init = np.array([0, 0, 0, 0, 0, 0], float)           # 3 DOF example
    eta_d = np.array([25/2-0.75-1, 0, 0, 0, 0, 0], float)
    mpc_config = {"N": N,
                  "dt": 1/control_fps,
                  "Q": np.diag([1, 1, 1]),
                  "R": np.diag([1, 1])}

    # Initialize vehicle and control
    vehicle = DubinsCar(dt=1/sim_fps)
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


def test_gradient():
    dt = 0.2
    N = 1

    conventional = False
    rl = True
    num_steps = 1
    plot_bool = False

    # harbour_geometry = [[-.25,  -.25],
    #                     [10.25,  -.25],
    #                     [10.25, 10.25],
    #                     [-.25, 10.25]]
    harbour_geometry = [[10, -15],
                        [11.75, -5],
                        [11.75, 5],
                        [10, 15],
                        [-12.5, 15],
                        [-12.5, -15]]
    harbour_space = utils.V2C(harbour_geometry)
    config = {
        "N": N,
        "dt": 0.2,
        "Q": np.diag([1, 10, 50]).tolist(),
        "q_slack": [100, 100, 100, 100, 100, 100],
        "R": np.diag([0.01, 0.01]).tolist(),
        "delta": 10,
        "q_xy": 20,
        "q_psi": 100,
        "gamma": 0.95,
        "alpha": 0.01,  # RL Learning rate
        "beta": 0.01  # SYSID Learning rate
    }

    u_init = 0.001*np.zeros(2)
    x_init = np.array([-5, 5, 0, 0, 0, 0])
    x_desired = np.array([25/2-0.75-0.5, 0, -np.pi/2])

    model = OtterModel(dt=dt, N=N)

    theta_init = model.theta

    opti = Optimizer()
    # x, u, s, theta, J_Q, model_constraint, xc, dual, lam_c = model.Q_step(x_init, u_init,
    #                                                                       x_desired, theta_init, config,
    #                                                                       opti, harbour_space)

    x, u, s = model._init_opt(x_init, u_init, opti)

    theta = opti.parameter(16+3)
    opti.set_value(theta, theta_init)
    model.update(theta)

    MRB = ca.MX.zeros(3, 3)
    MRB[0, 0] = theta[0]
    MRB[1, 1] = theta[0]
    MRB[1, 2] = theta[0] * theta[2]
    MRB[2, 1] = MRB[1, 2]
    MRB[2, 2] = theta[1]
    Minv = ca.inv(MRB)

    e = 0.001
    thrust = ca.vertcat(theta[10] * ca.sqrt(u[0] + e) * u[0],
                        theta[11] * ca.sqrt(u[1] + e) * u[1])

    tau = ca.vertcat(thrust[0] + thrust[1],
                     0,
                     -model.l1 * thrust[0] - model.l2 * thrust[1])

    eta_dot = utils.opt.Rz(x_init[2]) @ x[3:6, -1]
    nu_dot = Minv @ tau

    step = ca.vertcat(eta_dot, nu_dot)

    x_next = x[:, 0] + dt * step

    constraint = x_next
    model_constraint = x_next == x[:, 1]
    opti.subject_to(model_constraint)

    J = x[:3, -1].T @ x_desired

    opti.minimize(J)

    opti.solver("ipopt")
    sol = opti.solve()
    print(f"model_constrain: {model_constraint}")

    dual = opti.dual(model_constraint)

    l = J - dual.T @ constraint

    l_grad = ca.gradient(l, theta)
    gradient = opti.value(l_grad)

    print(f"l_grad: {l_grad}")
    print(f"gradient: {gradient}")
    print(f"dual: {dual}")
    print(f"dual value: {opti.value(dual)}")
