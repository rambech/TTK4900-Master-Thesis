from gymnasium import spaces
import numpy as np
import pygame


class Vehicle():
    """
    Base class for vehicles
    """

    def __init__(self, seed=None, dt=0.02) -> None:
        self.dt = dt

        # Action space for the otter is ±100%
        psi_max = 2*np.pi   # [rad] Heading
        psi_min = -psi_max  # [rad]
        # [m/s] Linear velocity in x (surge) direction, body frame
        u_max = 3.08667
        u_min = -3.08667    # [m/s]
        # [m/s] Linear velocity in y (sway) direction, body frame
        v_max = u_max/2
        # [m/s] Just a guess, have no clue of what to put here
        v_min = -v_max
        r_max = psi_max     # [rad/s] Heading rate
        r_min = psi_min     # [rad/s]

        v_bn_max = [u_max, v_max]
        v_bn_min = [u_min, v_min]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), seed=seed)
        self.info = {"Name": "Vehicle"}
        self.limits = {"psi_max": 2*np.pi,
                       "psi_min": -2*np.pi,
                       "nu_max": v_bn_max + [r_max],
                       "nu_min":  v_bn_min + [r_min],
                       "u_max": [1.0, 1.0],
                       "u_min": [-1.0, -1.0]}

        self.nu = np.zeros((3, 1), float)
        self.eta = np.zeros((3, 1), float)
        self.u = np.zeros((2, 1), float)

        self.L = 0  # Vehicle length
        self.B = 0  # Vehicle beam

        self.scale = 0  # Vehicle scale

    def step():
        """
        Normal step method for simulation
        """
        pass

    def rl_step(
            self, eta: np.ndarray, nu: np.ndarray, prev_u: np.ndarray,
            action: np.ndarray, beta_c: float, V_c: float) -> (np.ndarray, np.ndarray):
        """
        Step method for RL purposes
        """
        pass
