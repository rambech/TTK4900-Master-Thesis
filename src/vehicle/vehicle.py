from gymnasium import spaces
import numpy as np
import pygame

from utils import R2D, N2S


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

        self.dof = 3    # Controllable degrees of freedom
        self.Binv = None  # Overwritten

        # Must be overwritten
        self.vessel_image = None

    def _init_model():
        pass

    def step():
        """
        Normal step method for simulation
        """
        pass

    def unconstrained_allocation(self, tau) -> np.ndarray:
        u_control = self.Binv @ tau

        return u_control

    def render(self, eta: np.ndarray, offset: tuple[float, float]):
        rotated_image = pygame.transform.rotate(
            self.vessel_image, -R2D(eta[-1])).convert_alpha()
        eta_s = N2S(eta, self.scale, offset)
        center = (eta_s[0], eta_s[1])
        rect = rotated_image.get_rect(center=center)

        return rotated_image, rect

    def corners(self, eta: np.ndarray) -> list:
        """
        Returns the corners of the vehicle given its position

        x_v^n = l * cos(psi + delta)
        y_v^n = l * sin(psi + delta)

        where psi is heading and delta is the angle between x_b
        and the corners

        Parameters
        ----------
            eta : np.ndarray
                Pose in {n}

        Returns
        -------
            corners : np.ndarray
                Outer corners of the vessel given in (x,y) in {n}
        """

        half_L = self.L/2
        half_B = self.B/2
        l = np.linalg.norm((half_L, half_B), 2)
        psi_1 = eta[-1] + np.arctan2(half_B, half_L)
        psi_2 = eta[-1] + np.arctan2(half_B, -half_L)
        psi_3 = eta[-1] + np.arctan2(-half_B, -half_L)
        psi_4 = eta[-1] + np.arctan2(-half_B, half_L)

        forward_starboard = (eta[0]+l*np.cos(psi_1), eta[1]+l*np.sin(psi_1))
        forward_port = (eta[0]+l*np.cos(psi_2), eta[1]+l*np.sin(psi_2))
        aft_port = (eta[0]+l*np.cos(psi_3), eta[1]+l*np.sin(psi_3))
        aft_starboard = (eta[0]+l*np.cos(psi_4), eta[1]+l*np.sin(psi_4))

        corners = [forward_starboard, forward_port, aft_port, aft_starboard]

        return corners

    def step(self, eta: np.ndarray, nu: np.ndarray, prev_u: np.ndarray,
             action: np.ndarray, beta_c: float, V_c: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Step method for RL purposes
        """
        pass

    def _normalise(self, u):
        action = np.zeros(2).astype(np.float32)
        for idx, n in enumerate(u):
            if n < 0:
                action[idx] = (n/111)
            else:
                action[idx] = (n/113)

        return action

    def _denormalise(self, action):
        u = np.zeros(2).astype(np.float32)
        for idx, a in enumerate(action):
            if a < 0:
                u[idx] = a*111
            else:
                u[idx] = a*113

        return u
