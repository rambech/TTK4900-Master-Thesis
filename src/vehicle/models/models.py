import numpy as np
import casadi as ca


class Model():
    """
    Base class for vehicle models
    """

    def __init__(self, dt, N) -> None:
        self.dt = dt
        self.N = N

    def _init_model(self) -> None:
        pass

    def _init_opt(self, x_init, u_init, opti: ca.Opti):
        pass

    def step(self, x, u) -> None:
        pass

    def single_shooting(self, x_init: ca.DM, u_init: ca.DM,
                        opti: ca.Opti, use_slack: bool = None, space: tuple = None) -> tuple[ca.DM, ca.DM, ca.DM]:

        if space is not None:
            x, u, s = self._init_opt(x_init, u_init, opti, space)
        else:
            x, u, s = self._init_opt(x_init, u_init, opti)

        # Fixed step Runge-Kutta 4 integrator
        if use_slack:
            self.RK4(x, u, self.N, opti, s)
        else:
            self.RK4(x, u, self.N, opti)

        return x, u, s

    def direct_collocation(self, x_init, u_init, x_d, config, opti: ca.Opti, space: np.ndarray = None):
        """
        Direct collocation method 

        Based on the work of Joel Andersson, Joris Gillis and Moriz Diehl at KU Leuven

        Links:
        https://github.com/casadi/casadi/blob/main/docs/examples/matlab/direct_collocation_opti.m
        and
        https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py

        Parameters
        ----------
            x_init : np.ndarray
                Initial 

        """
        pass

    def RK4(self, x: ca.Opti.variable, u: ca.Opti.variable,
            N: int, opti: ca.Opti, s: ca.Opti.variable = None) -> None:
        """
        Runga-Kutta 4 method for making discrete model constraints

        Parameters
        ----------
            x : ca.Opti.variable
                State variables to optimize
            u : ca.Opti.variable
                Control input variables to optimize
            N : int
                Number of RK steps
            opti : ca.Opti
                Optimizer object


        Returns
        -------
            None

        """

        if s is not None:
            for k in range(N):
                # Fixed step Runge-Kutta 4 integrator
                k1 = self.step(x[:, k],                  u[:, k])
                k2 = self.step(x[:, k] + self.dt/2 * k1, u[:, k])
                k3 = self.step(x[:, k] + self.dt/2 * k2, u[:, k])
                k4 = self.step(x[:, k] + self.dt * k3,   u[:, k])
                x_next = x[:, k] + self.dt/6 * (k1+2*k2+2*k3+k4)

                opti.subject_to(x[:, k+1] == x_next + s[:, k])

        else:
            for k in range(N):
                # Fixed step Runge-Kutta 4 integrator
                k1 = self.step(x[:, k],                  u[:, k])
                k2 = self.step(x[:, k] + self.dt/2 * k1, u[:, k])
                k3 = self.step(x[:, k] + self.dt/2 * k2, u[:, k])
                k4 = self.step(x[:, k] + self.dt * k3,   u[:, k])
                x_next = x[:, k] + self.dt/6 * (k1+2*k2+2*k3+k4)

                opti.subject_to(x[:, k+1] == x_next)

    def update(self, theta: np.ndarray) -> None:
        """
        Update model parameters
        Takes in model parameters from another place and updates

        Parameters
        -----------
            theta : np.ndarray
                Model, initial 

        Returns
        -------
            self
        """
        pass
