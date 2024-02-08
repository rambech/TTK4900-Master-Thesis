import casadi as ca
import numpy as np

# TODO: Make euclidean objective function
# TODO: Make quadratic objective function
# TODO: Make huber and pseudo-huber objective functions


class Optimizer(ca.Opti):
    def __init__(self, *args):
        super().__init__(*args)

    def norm(self, x: ca.Opti.variable):
        return ca.sqrt(ca.mtimes(x.T, x))

    def euclidean(self, x: ca.Opti.variable, x_d: ca.Opti.parameter, weight: np.ndarray):
        """
        Euclidean objective function

        """
        self.minimize(weight*self.norm(x - x_d))

    def time(self, N: int):
        """
        Minimum time objective function
        """
        T = self.variable()
        self.subject_to(T >= 0)
        self.minimize(T)
        self.set_initial(T, 1)

        return T/N

    def quadratic(self, x: ca.Opti.variable, x_d: ca.Opti.parameter, weight: np.ndarray):
        self.minimize((x - x_d).T.dot(weight.dot(x - x_d)))
