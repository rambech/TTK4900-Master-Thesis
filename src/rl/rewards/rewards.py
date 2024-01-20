import numpy as np
from numpy.linalg import inv, det
from utils import ssa


def norm(eta, eta_d, has_crashed, has_docked):
    if has_crashed:
        reward = -10000
    elif has_docked:
        reward = 10000
    else:
        r1 = -np.linalg.norm(eta_d[0:2] - eta[0:2], 2)
        r2 = -abs(eta[-1] - eta_d[-1])
        reward = 10*(r1 + r2)

    return reward


def r_psi_e(psi_e, pos_e):
    """
    r = Ce^{-1/(2*sigma^2) pos_e^2}

    Parameters
    ----------
    psi_e : float
        Heading error
    pos_e : np.ndarray
        Position error in x and y

    Returns
    -------
    r : float
        Gaussian heading reward
    """

    if np.linalg.norm(pos_e, 2) <= 5:
        sigma = np.pi/4  # [rad]
        C = 0.3            # Max. along axis reward

        return C*np.exp(-1/(2*sigma**2) * psi_e**2)
    else:
        return 0


def r_pos_e(pos_e: np.ndarray) -> np.ndarray:
    """
    Bivariate Gaussian reward function

    r = -1 + C * e^{-1/2 (x^2/var + y^2/var))

    Parameters
    ----------
    pos_e : iterable[x: arraylike, y: arraylike]
        Cartesian position error (x, y)

    Returns
    -------
    reward : float
        Gaussian reward
    """
    sigma = 5
    var = sigma**2
    C = 2
    reward = C*np.exp(-(pos_e[0]**2/var + pos_e[1]**2/var)/2) - 1
    return reward


def r_surge(obs):
    """
    Possibility to use surge rewards for docking
    """
    # If within 5 meters of desired position,
    # allow for smaller velocities
    # if np.linalg.norm(obs[0:2], 2) <= 5:
    #     if -2.58 < obs[3] < 2.58:
    #         return 0.1

    # else:
    #     # If 0.5 < u < 5 knots, give positive reward
    #     if 0.257 < obs[3] < 2.58:
    #         return 0.1

    # return 0
    # return - np.linalg.norm(obs[3:6], 3)
    if np.linalg.norm(obs[0:2], 2) <= 5:
        return 1
    elif 0.257 < obs[3] < 2.58:
        return 1
    else:
        return -1


def r_time():
    return -0.1


def r_gaussian(obs):
    """
    r = C1e^{-1/(2*sigma^2) x_e^2} + C1e^{-1/(2*sigma^2) y_e^2} + C2e^{-1/((2*sigma^2) y_e^2}

    """
    pos_e = np.array([obs[0], obs[1]])

    return r_pos_e(pos_e) + r_psi_e(obs[2], pos_e)


def r_euclidean(obs):
    """
    r = - 100 * norm((x, y), 2)
    """
    # Makes no sense to have a global heading reward!
    return - 100 * np.linalg.norm(obs[0:2], 2)  # - abs(obs[2])


def r_quay(obs):
    return - 100 * obs[6]  # + abs(obs[7]))


def r_come_closer(obs, prev_obs, step_count):
    r = 0
    difference = r_euclidean(obs) - r_euclidean(prev_obs)
    if np.linalg.norm(obs[0:2], 2) <= 5:
        r = (r_quay(obs) - r_quay(prev_obs))
    elif r_euclidean(obs) - r_euclidean(prev_obs) > 0 and step_count < 500:
        # sign = np.sign(difference)
        r = min(80, (difference)**2)
    return r


def r_heading(obs, psi):
    # Heading towards the goal position
    if np.linalg.norm(obs[0:2], 2) <= 5:
        delta_psi = obs[2]
    else:
        ang2d = np.arctan2(obs[1], obs[0]) - np.pi
        delta_psi = ssa(psi - ang2d)

    sigma = np.pi/8    # [rad]
    C = 1              # Max. along axis reward

    return C*np.exp(-1/(2*sigma**2) * delta_psi**2)


def r_in_area(in_area):
    return 0.1 if in_area else 0


def r_backwards(obs):
    return -0.05 if obs[3] < 0 else 0
