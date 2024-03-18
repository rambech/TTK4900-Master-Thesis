import numpy as np
import casadi as ca

# TODO: Add region constraints


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

    def single_shooting(self, x_init, u_init, opti: ca.Opti):
        x, u, s = self._init_opt(x_init, u_init, opti)

        # Fixed step Runge-Kutta 4 integrator
        self.RK4(x, u, self.N, opti)

        return x, u, s

    def direct_collocation(self, x_init: np.ndarray,
                           u_init: np.ndarray, opti: ca.Opti) -> tuple:
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
            N: int, opti: ca.Opti) -> None:
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

        for k in range(N):
            k1 = self.step(x[:, k],                  u[:, k])
            k2 = self.step(x[:, k] + self.dt/2 * k1, u[:, k])
            k3 = self.step(x[:, k] + self.dt/2 * k2, u[:, k])
            k4 = self.step(x[:, k] + self.dt * k3,   u[:, k])
            x_next = x[:, k] + self.dt/6 * (k1+2*k2+2*k3+k4)
            opti.subject_to(x[:, k+1] == x_next)


# TODO: Add region constraints


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

    def single_shooting(self, x_init, u_init, opti: ca.Opti) -> tuple[ca.DM, ca.DM, ca.DM]:
        x, u, s = self._init_opt(x_init, u_init, opti)

        # Fixed step Runge-Kutta 4 integrator
        self.RK4(x, u, self.N, opti)

        return x, u, s

    def direct_collocation(self, x_init: np.ndarray,
                           u_init: np.ndarray, opti: ca.Opti) -> tuple:
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
            N: int, opti: ca.Opti) -> None:
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

        for k in range(N):
            # kx_1, ku_1 = self.step(
            #     x[:, k],
            #     u[:, k-1])

            # kx_2, ku_2 = self.step(
            #     x[:, k] + self.dt/2 * kx_1,
            #     u[:, k-1] + self.dt/2 * ku_1)

            # kx_3, ku_3 = self.step(
            #     x[:, k] + self.dt/2 * kx_2,
            #     u[:, k-1] + self.dt/2 * ku_2)
            # kx_4, ku_4 = self.step(
            #     x[:, k] + self.dt * kx_3,
            #     u[:, k-1] + self.dt/2 * ku_3)

            # x_next = x[:, k] + self.dt/6 * (kx_1+2*kx_2+2*kx_3+kx_4)
            # opti.subject_to(x[:, k+1] == x_next)

            # u_next = u[:, k-1] + self.dt/6 * (ku_1+2*ku_2+2*ku_3+ku_4)
            # opti.subject_to(u[:, k] == u_next)

            # Fixed step Runge-Kutta 4 integrator
            k1 = self.step(x[:, k],                  u[:, k])
            k2 = self.step(x[:, k] + self.dt/2 * k1, u[:, k])
            k3 = self.step(x[:, k] + self.dt/2 * k2, u[:, k])
            k4 = self.step(x[:, k] + self.dt * k3,   u[:, k])
            x_next = x[:, k] + self.dt/6 * (k1+2*k2+2*k3+k4)
            opti.subject_to(x[:, k+1] == x_next)
