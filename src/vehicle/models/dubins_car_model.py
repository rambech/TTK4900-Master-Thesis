import numpy as np
import casadi as ca
from utils import D2R
from .models import Model


class DubinsCarModel(Model):
    def __init__(self, dt: float = 0.05, N: int = 40) -> None:
        super().__init__(dt, N)

    def _init_opt(self, x_init, u_init, opti: ca.Opti):
        # Declaring optimization variables
        # State variables
        x = opti.variable(3, self.N+1)
        x_pos = x[0, :]
        y_pos = x[1, :]
        theta = x[2, :]

        # Input variables
        u = opti.variable(2, self.N)
        v = u[0, :]
        phi = u[1, :]

        # Slack variables
        s = opti.variable(3, self.N+1)

        # Control signal and time constraint
        opti.subject_to(opti.bounded(-1, v, 2))
        opti.subject_to(opti.bounded(D2R(-15), phi, D2R(15)))

        # Boundary values
        # Initial conditions
        opti.subject_to(x_pos[0] == x_init[0])
        opti.subject_to(y_pos[0] == x_init[1])
        opti.subject_to(theta[0] == x_init[2])
        opti.subject_to(v[0] == u_init[0])
        opti.subject_to(phi[0] == u_init[1])

        # Initial guesses
        opti.set_initial(x_pos, x_init[0])
        opti.set_initial(y_pos, x_init[1])
        opti.set_initial(theta, x_init[2])
        opti.set_initial(v, u_init[0])
        opti.set_initial(phi, u_init[1])

        return x, u, s

    def step(self, x, prev_u, u):
        """
        MPC model step for DubinsCarModel

        Parameters
        ----------
        x : np.ndarray
            State vector, x = [x, y, theta]
        u : np.ndarray
            Control input, u[0] = v and u[1] = phi
        """

        return ca.vertcat(u[0]*ca.cos(x[2]),
                          u[0]*ca.sin(x[2]),
                          u[0]*u[1])

    def direct_collocation(self, x_init: np.ndarray, u_init: np.ndarray, opti: ca.Opti) -> tuple:
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
        xdot = self.step(x, u)
