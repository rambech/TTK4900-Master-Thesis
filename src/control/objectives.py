"""
File containing objective functions for numerical optimization 
for control purposes.
"""
import casadi as ca
import numpy as np


def euclidean(init: ca.Opti.variable, goal: ca.Opti.variable, weight: np.ndarray, opti: ca.Opti):
    opti.minimize(np.linalg.norm(init - goal, 2))


def time(N, opti: ca.Opti):
    T = opti.variable()
    opti.subject_to(T >= 0)
    opti.minimize(T)
    opti.set_initial(T, 1)

    return T/N


def quadratic():
    pass
