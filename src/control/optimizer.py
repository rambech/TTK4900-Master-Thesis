import casadi as ca
import numpy as np

# TODO: Make euclidean objective function
# TODO: Make quadratic objective function
# TODO: Make huber and pseudo-huber objective functions


class Optimizer(ca.Opti):
    def __init__(self, *args):
        super().__init__(*args)

    def euclidean(self, x: ca.Opti.variable, x_d: ca.Opti.parameter, slack: ca.Opti.variable = None):
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

        weight = np.diag([1, 1, 1])

        if slack is not None:
            ...
        else:
            self.minimize(weight[0, 0]*ca.sum1(x[0]-x_d[0]) +
                          weight[1, 1]*ca.sum1(x[1]-x_d[1]) +
                          weight[2, 2]*ca.sum1(x[2]-x_d[2]))

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

    def simple_quadratic(self, x: ca.Opti.variable, x_d: ca.Opti.parameter, config: dict, slack: ca.Opti.variable = None):
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

        if slack is not None:
            ...
        else:
            self.minimize(config["Q"][0, 0]*(x[0, -1]-x_d[0, -1])**2 +
                          config["Q"][1, 1]*(x[1, -1]-x_d[1, -1])**2 +
                          config["Q"][2, 2]*(x[2, -1]-x_d[2, -1])**2)

    def quadratic(self, x: ca.Opti.variable, u: ca.Opti.variable, x_d: ca.Opti.parameter, config: dict, slack: ca.Opti.variable = None):
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

        if slack is not None:
            ...
        else:
            # TODO: Fix, works a bit weirdly
            self.minimize(config["Q"][0, 0]*ca.sum2(x[0, 1:]-x_d[0, 1:])**2 +
                          config["Q"][1, 1]*ca.sum2(x[1, 1:]-x_d[1, 1:])**2 +
                          config["Q"][2, 2]*ca.sum2(x[2, 1:]-x_d[2, 1:])**2)
            #   config["R"][0, 0]*ca.sum2(u[0])**2 +
            #   config["R"][1, 1]*ca.sum2(u[1])**2)
