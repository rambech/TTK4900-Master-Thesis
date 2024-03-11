import casadi as ca
import numpy as np

# TODO: Make euclidean objective function
# TODO: Make quadratic objective function
# TODO: Make huber and pseudo-huber objective functions


class Optimizer(ca.Opti):
    def __init__(self, *args):
        super().__init__(*args)

        self._default_config = {"N": 40,
                                "dt": 0.05,
                                "Q": np.diag([1, 1, 1]),
                                "Q_slack": np.diag([1, 1, 1]),
                                "R": np.diag([1, 1])}

    def euclidean(self, x: ca.Opti.variable, x_d: ca.Opti.parameter,
                  config: dict = None, slack: ca.Opti.variable = None):
        """
        Euclidean objective function

        min norm(x, 2)

        Parameters
        ----------
        x : ca.Opti.variable
            State variables
        x_d : ca.Opti.parameter
            Desired end state
        slack : ca.Opti.variable
            Slack variable
        """

        if not config:
            config = self._default_config

        if slack is not None:
            ...
        else:
            self.minimize(config["Q"][0, 0]*ca.sum1(x[0]-x_d[0]) +
                          config["Q"][1, 1]*ca.sum1(x[1]-x_d[1]) +
                          config["Q"][2, 2]*ca.sum1(x[2]-x_d[2]))

    def time(self, N: int):
        """
        Minimum time objective function

        min T

        Parameters
        ----------
        N : int
            Time horizon
        """

        T = self.variable()
        self.subject_to(T >= 0)
        self.minimize(T)
        self.set_initial(T, 1)

        return T/N

    def simple_quadratic(self, x: ca.Opti.variable, x_d: ca.Opti.parameter,
                         config: dict = None, slack: ca.Opti.variable = None):
        """
        Simple quadratic objective function

        min (x_t - x_d)^2 + (y_t - y_d)^2 + (psi_t - psi_d)^2 

        Parameters
        ----------
        x : ca.Opti.variable
            State variables
        x_d : ca.Opti.parameter
            Desired end state
        config : dict
            Penalty weights for objective function
        slack : ca.Opti.variable
            Slack variable
        """

        if not config:
            config = self._default_config

        if slack is not None:
            self.minimize(config["Q"][0, 0]*(x[0, -1]-x_d[0, -1])**2 +
                          config["Q"][1, 1]*(x[1, -1]-x_d[1, -1])**2 +
                          config["Q"][2, 2]*(x[2, -1]-x_d[2, -1])**2 +
                          config["Q_slack"][0, 0]*slack[0, -1]**2 +
                          config["Q_slack"][1, 1]*slack[1, -1]**2 +
                          config["Q_slack"][2, 2]*slack[2, -1]**2)
        else:
            self.minimize(config["Q"][0, 0]*(x[0, -1]-x_d[0, -1])**2 +
                          config["Q"][1, 1]*(x[1, -1]-x_d[1, -1])**2 +
                          config["Q"][2, 2]*(x[2, -1]-x_d[2, -1])**2)

    def quadratic(self, x: ca.Opti.variable, u: ca.Opti.variable, x_d: ca.Opti.parameter,
                  config: dict = None, slack: ca.Opti.variable = None):
        """
        Simple quadratic objective function

        min norm(x, 2)^2 + norm(u, 2)^2 

        Parameters
        ----------
        x : ca.Opti.variable
            State variables
        x_d : ca.Opti.parameter
            Desired end state
        config : dict
            Penalty weights for objective function
        slack : ca.Opti.variable
            Slack variable
        """

        if not config:
            config = self._default_config

        if slack is not None:
            ...
        else:
            # TODO: Fix, works a bit weirdly
            self.minimize(config["Q"][0, 0]*ca.sum2(x[0, 1:]-x_d[0, 1:])**2 +
                          config["Q"][1, 1]*ca.sum2(x[1, 1:]-x_d[1, 1:])**2 +
                          config["Q"][2, 2]*ca.sum2(x[2, 1:]-x_d[2, 1:])**2 +
                          config["R"][0, 0]*ca.sum2(u[0])**2 +
                          config["R"][1, 1]*ca.sum2(u[1])**2)

    def pseudo_huber(self, x: ca.Opti.variable, u: ca.Opti.variable,
                     x_d: ca.Opti.parameter, config: dict = None):
        """
        Full objective function utilizing pseudo-Huber

        min q_xy * f_xy(eta_N, eta_d) + q_psi * f_psi(eta_N, eta_d) 
            + sum(nu.T.dot(Q.dot(nu)) + tau.T.dot(R.dot(tau)))

        """

        if not config:
            config = self._default_config

        self.minimize(config["q_xy"]*self._pos_pseudo_huber(x, x_d) +
                      config["q_psi"]*self._heading_cost(x, x_d) +
                      ca.sum2(x[3:6].T @ config["Q"] @ x[3:6]) +
                      ca.sum2(u.T @ config["R"] @ u))

    def rl(self):
        ...

    def _pos_pseudo_huber(self, x, x_d, delta):
        """
        Position pseudo-Huber cost

        f_xy(eta_N, eta_d) = delta**2 (sqrt(1 + ((x - x_d)**2 + (y - y_d)**2) / delta**2) - 1)

        """

        return delta**2 * (ca.sqrt(1 + ((x[0, -1] - x_d[0, -1])**2 +
                                        (x[1, -1] - x_d[1, -1])**2) / delta**2) - 1)

    def _heading_cost(self, x, x_d, delta):
        """
        Heading reward

        f_psi(eta_N, eta_d) = (1 - cos(psi - psi_d))/2

        """

        return (1 - ca.cos(x[2, -1] - x_d[2, -1]))/2
