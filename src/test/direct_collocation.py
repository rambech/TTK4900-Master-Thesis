import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from vehicle.models import DubinsCarModel
from plotting import plot_solution
from control.optimizer import Optimizer
from utils import D2R


def direct_collocation_example():

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

    x_d = ca.vertcat(15,
                     15,
                     0)

    # Setup inputs
    v = ca.SX.sym("v")
    phi = ca.SX.sym("phi")
    u = ca.vertcat(v,
                   phi)

    # Control discretization
    # N = 40  # number of control intervals
    # dt = T/N
    N = 50
    dt = 0.05

    model = DubinsCarModel(dt=dt, N=N)

    # Model
    xdot = model.step(x, u)

    # Objective function
    L = (x_pos - x_d[0])**2 + (y_pos - x_d[1])**2 + \
        (theta - x_d[2])**2  # + v**2 + phi**2

    # Continuous time dynamics
    f = ca.Function('f', [x, u], [xdot, L], ['x', 'u'], ['xdot', 'L'])

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
    lbw.append([0, 0, 0])
    ubw.append([0, 0, 0])
    w0.append([0, 0, 0])
    x_plot.append(Xk)

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = ca.MX.sym('U_' + str(k), 2)
        w.append(Uk)
        lbw.append([-1, D2R(-15)])
        ubw.append([1, D2R(15)])
        w0.append([0, 0])
        u_plot.append(Uk)

        # State at collocation points
        Xc = []
        for j in range(d):
            Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j), 3)
            Xc.append(Xkj)
            w.append(Xkj)
            lbw.append([-np.inf, -np.inf, -np.inf])
            ubw.append([np.inf,  np.inf, np.inf])
            w0.append([0, 0, 0])

        # Loop over collocation points
        Xk_end = D[0]*Xk
        for j in range(1, d+1):
            # Expression for the state derivative at the collocation point
            xp = C[0, j]*Xk
            for r in range(d):
                xp = xp + C[r+1, j]*Xc[r]

            # Append collocation equations
            fj, qj = f(Xc[j-1], Uk)
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
        lbw.append([-np.inf, -np.inf, -np.inf])
        ubw.append([np.inf,  np.inf, np.inf])
        w0.append([0, 0, 0])
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
    solver = ca.nlpsol('solver', 'ipopt', prob)

    # Function to get x and u trajectories from w
    trajectories = ca.Function('trajectories', [w], [
        x_plot, u_plot], ['w'], ['x', 'u'])

    # Solve the NLP
    solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    x_opt, u_opt = trajectories(solution['x'])
    x_opt = x_opt.full()  # to numpy array
    u_opt = u_opt.full()  # to numpy array

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


def dubins_distance_direct_collocation_example():
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
    f = ca.Function("f", [x, u], [xdot, L], ["x", "u"], ["xdot", "L"])

    # Initialize empty NLP
    opti = Optimizer()
    J = 0

    # Don't know what it means but "lift" initial conditions
    Xk = opti.variable(3)
    opti.subject_to(Xk == ca.vertcat(0, 0, 0))
    opti.set_initial(Xk, ca.vertcat(0, 0, 0))

    # Apparently collect all states/controls
    Xs = Xk
    Us = []

    # Formulate the NLP
    for k in range(N):
        # New NLP variable for control
        Uk = opti.variable(2)
        if k == 0:
            Us = Uk
        else:
            Us = ca.horzcat(Us, Uk)

        opti.subject_to(opti.bounded(-1, Uk[0], 1))
        opti.subject_to(opti.bounded(D2R(-15), Uk[1], D2R(15)))
        opti.set_initial(Uk[0], 0)
        opti.set_initial(Uk[1], 0)

        # Decision variables for helper states at each collocation point
        Xc = opti.variable(3, d)

        # Position constraint
        opti.subject_to(opti.bounded(-np.inf, Xc[0, :], np.inf))
        opti.subject_to(opti.bounded(-np.inf, Xc[1, :], np.inf))
        opti.subject_to(opti.bounded(-np.inf, Xc[2, :], np.inf))
        opti.set_initial(Xc, np.tile([0, 0, 0], (d, 1)))

        # Evaluate ODE right-hand-side at all helper states
        ode, quad = f(Xc, Uk)

        for j in range(1, d+1):
            # Add contribution to quadrature function
            J += quad[j-1]*B[j]*dt

        # Get interpolating points of collocation polynomial
        Z = ca.horzcat(Xk, Xc)
        # print(f"ode.shape {ode.shape}")

        # Get slope of interpolating polynomial (normalized)
        Pidot = Z @ C[:, 1:]

        # Match with ODE right-hand-side
        opti.subject_to(Pidot == dt*ode)

        # State at end of collocation interval
        Xk_end = Z @ D

        # New decision variable for state at end of interval
        Xk = opti.variable(3)
        # Xs.append(Xk)
        Xs = ca.horzcat(Xs,
                        Xk)
        opti.subject_to(-0.25 <= Xk[0])
        opti.set_initial(Xk, [0, 0, 0])

        # Continuity constraints
        opti.subject_to(Xk_end == Xk)

    # Setup solver and solve
    opti.solver('ipopt')
    solution = opti.solve()

    print(f"Xs_opt {np.round(solution.value(Xs))}")
    print(f"Us_opt {np.round(solution.value(Us))}")

    plot_solution(solution, Xs, Us)
    # TODO: Finnish this python/direct collocation/opti example
