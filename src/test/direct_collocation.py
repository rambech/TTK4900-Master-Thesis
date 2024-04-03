import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
from vehicle.models import DubinsCarModel, OtterModel
from plotting import plot_solution
from control.optimizer import Optimizer
from utils import D2R, kts2ms


def direct_collocation_example(x_init: np.ndarray = np.zeros(3),
                               u_init: np.ndarray = np.zeros(2),
                               give_value: bool = False,
                               plot: bool = True):
    N = 50
    dt = 0.05

    model = DubinsCarModel(dt=dt, N=N)

    # Degree of interpolating polynomial
    d = 3

    # Get collocation points
    tau_root = np.append(0, ca.collocation_points(d, 'legendre'))

    # Coefficients of the collocation equation
    C = np.zeros((d+1, d+1))

    # Coefficients of the continuity equation
    D = np.zeros(d+1)

    # Coefficients of the quadrature function
    B = np.zeros(d+1)

    # Construct polynomial basis
    for j in range(d+1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d+1):
            C[j, r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    # Time horizon
    T = 10.

    # Declare model variables
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

    x_d = ca.vertcat(10,
                     10,
                     0)

    # Model
    # xdot = model.step(x, u)

    # Objective function
    L = (x_pos - x_d[0])**2 + (y_pos - x_d[1])**2 + \
        (theta - x_d[2])**2  # + v**2 + phi**2

    # Continuous time dynamics
    # f = ca.Function('f', [x, u], [xdot, L], ['x', 'u'], ['xdot', 'L'])
    f = ca.Function('f', [x, u], [L], ['x', 'u'], ['L'])

    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []

    # For plotting x and u given w
    x_plot = []
    u_plot = []

    # "Lift" initial conditions
    Xk = ca.MX.sym('X0', 3)
    w.append(Xk)
    lbw.append([x_init[0], x_init[1], x_init[2]])
    ubw.append([x_init[0], x_init[1], x_init[2]])
    w0.append([x_init[0], x_init[1], x_init[2]])
    x_plot.append(Xk)

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = ca.MX.sym('U_' + str(k), 2)
        w.append(Uk)
        # lbw.append([-1, D2R(-15)])
        # ubw.append([1, D2R(15)])
        # w0.append([0, 0])

        # First control input
        if k == 0:
            lbw.append([u_init[0], u_init[1]])
            ubw.append([u_init[0], u_init[1]])
            w0.append([u_init[0], u_init[1]])
        else:
            lbw.append([-1, D2R(-15)])
            ubw.append([1, D2R(15)])
            w0.append([u_init[0], u_init[1]])

        u_plot.append(Uk)

        # State at collocation points
        Xc = []
        for j in range(d):
            Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j), 3)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append([-0.25, -0.25, -np.inf])
            ubw.append([10.25,  10.25, np.inf])
            w0.append([x_init[0], x_init[1], x_init[2]])

        # Loop over collocation points
        Xk_end = D[0]*Xk
        for j in range(1, d+1):
            # Expression for the state derivative at the collocation point
            xp = C[0, j]*Xk
            for r in range(d):
                xp = xp + C[r+1, j]*Xc[r]

            # Append collocation equations
            # fj, qj = f(Xc[j-1], Uk)
            qj = f(Xc[j-1], Uk)
            fj = model.step(Xc[-1], Uk)
            g.append(dt*fj - xp)
            lbg.append([0, 0, 0])
            ubg.append([0, 0, 0])

            # Add contribution to the end state
            Xk_end = Xk_end + D[j]*Xc[j-1]

            # Add contribution to quadrature function
            J = J + B[j]*qj*dt

        # New NLP variable for state at end of interval
        Xk = ca.MX.sym('X_' + str(k+1), 3)
        w.append(Xk)
        lbw.append([-0.25, -0.25, -np.inf])
        ubw.append([10.25,  10.25, np.inf])
        w0.append([x_init[0], x_init[1], x_init[2]])
        x_plot.append(Xk)

        # Add equality constraint
        g.append(Xk_end-Xk)
        lbg.append([0, 0, 0])
        ubg.append([0, 0, 0])

    # Concatenate vectors
    w = ca.vertcat(*w)
    g = ca.vertcat(*g)
    x_plot = ca.horzcat(*x_plot)
    u_plot = ca.horzcat(*u_plot)
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    # Create an NLP solver
    prob = {'f': J, 'x': w, 'g': g}
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    solver = ca.nlpsol('solver', 'ipopt', prob, opts)

    # Function to get x and u trajectories from w
    trajectories = ca.Function('trajectories', [w], [
        x_plot, u_plot], ['w'], ['x', 'u'])

    # Solve the NLP
    solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    x_opt, u_opt = trajectories(solution['x'])
    x_opt = x_opt.full()  # to numpy array
    u_opt = u_opt.full()  # to numpy array

    if plot:
        print("Plotting")
        # Position plot
        fig0, ax0 = plt.subplots(figsize=(7, 7))
        ax0.plot(x_opt[0, :], x_opt[1, :])
        ax0.set(xlabel="x position", ylabel="y position")

        t_data = np.linspace(0, len(u_opt), num=len(u_opt[0]))
        # U
        fig4, ax4 = plt.subplots(figsize=(7, 7))
        ax4.step(t_data, u_opt[0, :],  where="post")
        ax4.step(t_data, u_opt[1, :], where="post")
        ax4.set(xlabel="t", ylabel="u control")

        # Plot the result
        # tgrid = np.linspace(0, T, N+1)
        # plt.figure(1)
        # plt.clf()
        # plt.plot(tgrid, x_opt[0], '--')
        # plt.plot(tgrid, x_opt[1], '-')
        # plt.step(tgrid, np.append(np.nan, u_opt[0]), '-.')
        # plt.xlabel('t')
        # plt.legend(['x1', 'x2', 'u'])
        # plt.grid()
        plt.show()

    if give_value:
        return x_opt, u_opt


def opti_direct_collocation_example(x_init: np.ndarray = np.zeros(3),
                                    u_init: np.ndarray = np.zeros(2),
                                    give_value: bool = False,
                                    plot: bool = True):
    """
    Direct collocation method 

    Based on the work of Joel Andersson, Joris Gillis and Moriz Diehl at KU Leuven

    Links:
    https://github.com/casadi/casadi/blob/main/docs/examples/matlab/direct_collocation_opti.m
    and
    https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py


    """

    N = 50     # Control intervals
    dt = 0.05   # Time step length

    model = DubinsCarModel(dt=dt, N=N)

    # Degree of interpolating polynomial
    d = 3

    # Get collocation points
    tau_root = np.append(0, ca.collocation_points(d, "legendre"))

    # Collocation, continuity and quadrature coefficients
    C, D, B = np.zeros((d+1, d+1)), np.zeros(d+1), np.zeros(d+1)

    for j in range(d+1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

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

    # Desired pose
    x_d = ca.vertcat(10,
                     10,
                     0)

    # Model
    xdot = model.step(x, u)

    # Objective function
    L = (x_pos - x_d[0])**2 + (y_pos - x_d[1])**2 + \
        (theta - x_d[2])**2  # + v**2 + phi**2

    # Continuous time dynamics
    f = ca.Function('f', [x, u], [xdot, L], ['x', 'u'], ['xdot', 'L'])

    # Initialize empty NLP
    opti = Optimizer()
    J = 0

    # Don't know what it means but "lift" initial conditions
    # Xk = opti.variable(3)
    # opti.subject_to(Xk == ca.vertcat(0, 0, 0))
    # opti.set_initial(Xk, ca.vertcat(0, 0, 0))

    # Apparently collect all states/controls
    # Xs = Xk
    # Us = []
    x_init = ca.vertcat(x_init[0],
                        x_init[1],
                        x_init[2])
    Xs = opti.variable(3, N+1)
    opti.subject_to(Xs[:, 0] == x_init)
    # opti.set_initial(Xs[:, 0], ca.vertcat(0, 0, 0))
    opti.subject_to(opti.bounded(-0.25, Xs[0, 1:], 10.25))
    opti.subject_to(opti.bounded(-0.25, Xs[1, 1:], 10.25))
    opti.subject_to(opti.bounded(-np.inf, Xs[2, 1:], np.inf))
    # opti.set_initial(Xs, np.zeros((3, N+1)))
    opti.set_initial(Xs, np.tile(x_init, (1, N+1)))

    Us = opti.variable(2, N)
    opti.subject_to(opti.bounded(-1, Us[0, :], 1))
    opti.subject_to(opti.bounded(D2R(-15), Us[1, :], D2R(15)))
    opti.set_initial(Us[0, :], u_init[0])
    opti.set_initial(Us[1, :], u_init[1])

    # TODO: Is this possible instead?
    # opti.set_initial(Us, u_init)

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for control
        # Uk = opti.variable(2)
        # if k == 0:
        #     Us = Uk
        # else:
        #     Us = ca.horzcat(Us, Uk)

        # opti.subject_to(opti.bounded(-1, Uk[0], 1))
        # opti.subject_to(opti.bounded(D2R(-15), Uk[1], D2R(15)))
        # opti.set_initial(Uk[0], 0)
        # opti.set_initial(Uk[1], 0)

        # Decision variables for helper states at each collocation point
        Xc = opti.variable(3, d)

        # Position constraint
        opti.subject_to(opti.bounded(-np.inf, Xc[0, :], np.inf))
        opti.subject_to(opti.bounded(-np.inf, Xc[1, :], np.inf))
        opti.subject_to(opti.bounded(-np.inf, Xc[2, :], np.inf))
        opti.set_initial(Xc, np.tile(x_init, (1, d)))

        # Evaluate ODE right-hand-side at all helper states
        # ode, quad = f(Xc, Uk)
        # ode, quad = f(Xc, Us[:, k])

        # Xk_end = D[0]*Xs[:, k]
        for j in range(1, d+1):
            #     # Expression for the state derivative at the collocation point
            #     xp = C[0, j]*Xs[:, k]
            #     for r in range(d):
            #         xp = xp + C[r+1, j]*Xc[r]

            #     # Append collocation equations
            ode, quad = f(Xc[j-1], Us[:, k])
        #     opti.subject_to(xp == dt*ode)

        #     # Add contribution to the end state
        #     Xk_end = Xk_end + D[j]*Xc[j-1]

        #     # Add contribution to quadrature function
            J = J + quad*B[j]*dt

        # J = J + quad*B*dt

        # Get interpolating points of collocation polynomial
        Z = ca.horzcat(Xs[:, k], Xc)
        # # print(f"ode.shape {ode.shape}")

        # # Get slope of interpolating polynomial (normalized)
        Pidot = Z @ C[:, 1:]

        # # Match with ODE right-hand-side
        opti.subject_to(Pidot == dt*ode)

        # State at end of collocation interval
        Xk_end = Z @ D

        # New decision variable for state at end of interval
        # Xk = opti.variable(3)
        # # Xs.append(Xk)
        # Xs = ca.horzcat(Xs,
        #                 Xk)
        # opti.subject_to(-0.25 <= Xs[0, k+1])
        # opti.set_initial(Xs[:, k+1], [0, 0, 0])

        # Continuity constraints
        opti.subject_to(Xk_end == Xs[:, k+1])

    opti.minimize(J)

    # Setup solver and solve
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', opts)
    solution = opti.solve()

    print(f"Xs_opt {np.round(solution.value(Xs))}")
    print(f"Us_opt {np.round(solution.value(Us))}")

    if plot:
        plot_solution(solution, Xs, Us)

    if give_value:
        return solution.value(Xs), solution.value(Us)


def mpc_direct_collocation_example():
    x, u = np.zeros(3), np.zeros(2)
    x_opti, u_opti = x.copy(), u.copy()
    time_list = []
    opti_time_list = []
    for i in range(1, 51):
        if i % 10 == 0:
            plot = True
        else:
            plot = False

        t0 = time.time()
        x_opt, u_opt = direct_collocation_example(x, u, True, plot)
        t1 = time.time()

        t = t1 - t0
        time_list.append(t)

        x, u = x_opt[:, 1], u_opt[:, 1]

        t0_opti = time.time()
        x_opt_opti, u_opt_opti = opti_direct_collocation_example(
            x_opti, u_opti, True, plot)
        t1_opti = time.time()

        t_opti = t1_opti - t0_opti
        opti_time_list.append(t_opti)

        x_opti, u_opti = x_opt_opti[:, 1], u_opt_opti[:, 1]

        print(f"x: {x}")
        print(f"u: {u}")
        print(f"x_opti: {x_opti}")
        print(f"u_opti: {u_opti}")

    print(f"Average time used: {np.mean(time_list)}")
    print(f"Max time used: {np.max(time_list)}")
    print(f"Min time used: {np.min(time_list)}")
    print(f"Average opti time used: {np.mean(opti_time_list)}")
    print(f"Max opti time used: {np.max(opti_time_list)}")
    print(f"Min opti time used: {np.min(opti_time_list)}")


def otter_direct_collocation_example(x_init=np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001]).T,
                                     u_init=np.array([0.001, 0.001]).T,
                                     give_value: bool = False,
                                     plot: bool = True):
    t1 = time.time()
    N = 50
    dt = 0.2

    model = OtterModel(dt=dt, N=N)

    # Degree of interpolating polynomial
    d = 3

    # Get collocation points
    tau_root = np.append(0, ca.collocation_points(d, 'legendre'))

    # Coefficients of the collocation equation
    C = np.zeros((d+1, d+1))

    # Coefficients of the continuity equation
    D = np.zeros(d+1)

    # Coefficients of the quadrature function
    B = np.zeros(d+1)

    # Construct polynomial basis
    for j in range(d+1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d+1):
            C[j, r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    # Time horizon
    T = 10.

    # Declare model variables
    north = ca.SX.sym("north")
    east = ca.SX.sym("east")
    yaw = ca.SX.sym("yaw")
    surge = ca.SX.sym("surge")
    sway = ca.SX.sym("sway")
    yaw_rate = ca.SX.sym("yaw_rate")
    x = ca.vertcat(north,
                   east,
                   yaw,
                   surge,
                   sway,
                   yaw_rate)

    # Setup inputs
    n_port = ca.SX.sym("n_port")
    n_stb = ca.SX.sym("n_stb")
    u = ca.vertcat(n_port,
                   n_stb)

    x_d = ca.vertcat(10,
                     10,
                     0)

    # Model
    # xdot = model.step(x, u)

    # Objective function
    L = (north - x_d[0])**2 + (east - x_d[1])**2 + \
        (yaw - x_d[2])**2  # + v**2 + phi**2

    f = ca.Function("f", [x, u], [L], ["x", "u"], ["L"])
    # Continuous time dynamics
    # f = ca.Function('f', [x, u], [xdot, L], ['x', 'u'], ['xdot', 'L'])

    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []

    # For plotting x and u given w
    x_plot = []
    u_plot = []

    # "Lift" initial conditions
    Xk = ca.MX.sym('X0', 6)
    w.append(Xk)
    lbw.append([x_init[0], x_init[1], x_init[2],
                x_init[3], x_init[4], x_init[5]])
    ubw.append([x_init[0], x_init[1], x_init[2],
                x_init[3], x_init[4], x_init[5]])
    w0.append([x_init[0], x_init[1], x_init[2],
               x_init[3], x_init[4], x_init[5]])
    x_plot.append(Xk)

    # Formulate the NLP
    for k in range(N):
        if k == 0:
            # First control input must be u_init
            Uk = ca.MX.sym('U_' + str(k), 2)
            w.append(Uk)
            lbw.append([u_init[0], u_init[1]])
            ubw.append([u_init[0], u_init[1]])
            w0.append([u_init[0], u_init[1]])
            u_plot.append(Uk)
        else:
            # New NLP variable for the control
            Uk = ca.MX.sym('U_' + str(k), 2)
            w.append(Uk)
            # print(f"len w in else statement: {len(w)}")
            lbw.append([-70, -70])
            ubw.append([100, 100])
            w0.append([u_init[0], u_init[1]])
            u_plot.append(Uk)

            k_prev = len(w)-6
            g.append(Uk - w[k_prev])
            lbg.append([-100*dt, -100*dt])
            ubg.append([100*dt, 100*dt])
            # TODO: Add delta constraint here

        # State at collocation points
        Xc = []
        for j in range(d):
            Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j), 6)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append([-1, -1, -np.inf, kts2ms(-5), kts2ms(-5), -np.pi])
            ubw.append([10.25, 10.25, np.inf, kts2ms(5), kts2ms(5), np.pi])
            w0.append([x_init[0], x_init[1], x_init[2],
                       x_init[3], x_init[4], x_init[5]])

        # Loop over collocation points
        Xk_end = D[0]*Xk
        for j in range(1, d+1):
            # Expression for the state derivative at the collocation point
            xp = C[0, j]*Xk
            for r in range(d):
                xp = xp + C[r+1, j]*Xc[r]

            # Append collocation equations
            # fj, qj = f(Xc[j-1], Uk)
            fj = model.step(Xc[j-1], Uk)
            qj = f(Xc[j-1], Uk)

            g.append(dt*fj - xp)
            lbg.append([0, 0, 0, 0, 0, 0])
            ubg.append([0, 0, 0, 0, 0, 0])

            # Add contribution to the end state
            Xk_end = Xk_end + D[j]*Xc[j-1]

            # Add contribution to quadrature function
            J = J + B[j]*qj*dt

        # New NLP variable for state at end of interval
        Xk = ca.MX.sym('X_' + str(k+1), 6)
        w.append(Xk)
        lbw.append([-1, -1, -np.inf, kts2ms(-5), kts2ms(-5), -np.pi])
        ubw.append([10.25, 10.25, np.inf, kts2ms(5), kts2ms(5), np.pi])
        w0.append([x_init[0], x_init[1], x_init[2],
                   x_init[3], x_init[4], x_init[5]])
        x_plot.append(Xk)

        # Add equality constraint
        g.append(Xk_end-Xk)
        lbg.append([0, 0, 0, 0, 0, 0])
        ubg.append([0, 0, 0, 0, 0, 0])

    # Concatenate vectors
    w = ca.vertcat(*w)
    g = ca.vertcat(*g)
    x_plot = ca.horzcat(*x_plot)
    u_plot = ca.horzcat(*u_plot)
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    # Create an NLP solver
    prob = {'f': J, 'x': w, 'g': g}
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.warm_start_init_point': 'yes'}
    # 'ipopt.warm_start_bound_push': 1e-6,
    # 'ipopt.warm_start_mult_bound_push': 1e-6
    # opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    solver = ca.nlpsol('solver', 'ipopt', prob, opts)

    # Function to get x and u trajectories from w
    trajectories = ca.Function('trajectories', [w], [
        x_plot, u_plot], ['w'], ['x', 'u'])

    # Solve the NLP
    solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    x_opt, u_opt = trajectories(solution['x'])
    x_opt = np.round(x_opt.full(), 2)  # to numpy array
    u_opt = np.round(u_opt.full(), 2)  # to numpy array
    t2 = time.time()

    _time = t2 - t1

    print(f"Time: {_time}")
    if plot:
        # Position plot
        fig0, ax0 = plt.subplots(figsize=(7, 7))
        ax0.plot(x_opt[1, :], x_opt[0, :])
        ax0.set(xlabel="East", ylabel="North")

        t_data = np.linspace(0, len(u_opt), num=len(u_opt[0]))
        # U
        fig4, ax4 = plt.subplots(figsize=(7, 7))
        ax4.step(t_data, u_opt[0, :],  where="post")
        ax4.step(t_data, u_opt[1, :], where="post")
        ax4.set(xlabel="t", ylabel="u control")

        # Plot the result
        # tgrid = np.linspace(0, T, N+1)
        # plt.figure(1)
        # plt.clf()
        # plt.plot(tgrid, x_opt[0], '--')
        # plt.plot(tgrid, x_opt[1], '-')
        # plt.step(tgrid, np.append(np.nan, u_opt[0]), '-.')
        # plt.xlabel('t')
        # plt.legend(['x1', 'x2', 'u'])
        # plt.grid()
        plt.show()

    if give_value:
        return x_opt, u_opt


def otter_mpc_direct_collocation_example():
    x, u = np.zeros(6), np.zeros(2)
    # x_opti, u_opti = x.copy(), u.copy()
    time_list = []
    # opti_time_list = []
    for i in range(1, 51):
        if i % 10 == 0:
            plot = False
        else:
            plot = False

        t0 = time.time()
        x_opt, u_opt = otter_direct_collocation_example(x, u, True, plot)
        t1 = time.time()

        t = t1 - t0
        time_list.append(t)

        x, u = x_opt[:, 1], u_opt[:, 1]

        # t0_opti = time.time()
        # x_opt_opti, u_opt_opti = opti_direct_collocation_example(
        #     x_opti, u_opti, True, plot)
        # t1_opti = time.time()

        # t_opti = t1_opti - t0_opti
        # opti_time_list.append(t_opti)

        # x_opti, u_opti = x_opt_opti[:, 1], u_opt_opti[:, 1]

        print(f"x: {x}")
        print(f"u: {u}")
        # print(f"x_opti: {x_opti}")
        # print(f"u_opti: {u_opti}")

    print(f"Average time used: {np.mean(time_list)}")
    print(f"Std time used: {np.std(time_list)}")
    print(f"Max time used: {np.max(time_list)}")
    print(f"Min time used: {np.min(time_list)}")
    # print(f"Average opti time used: {np.mean(opti_time_list)}")
    # print(f"Max opti time used: {np.max(opti_time_list)}")
    # print(f"Min opti time used: {np.min(opti_time_list)}")
