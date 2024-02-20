import numpy as np
from .vehicle import Vehicle
import pygame
from utils import N2B, sat, D2R


class DubinsCar(Vehicle):
    """
    Simplest possible vehicle model, based on Dubins Car
    with added controllable velocity
    https://lavalle.pl/planning/node821.html
    """

    def __init__(self, seed=None, dt=0.02) -> None:
        super().__init__(seed, dt)

        self._init_model()  # Override dummy model

        # For render
        self.scale = 30  # [px/m]
        self.vessel_image = pygame.image.load(
            'vehicle/images/car.png')
        self.vessel_image = pygame.transform.scale(
            self.vessel_image, (self.scale*2*self.B, self.scale*1.5*self.L))

    def _init_model(self):
        # Car size and constants
        self.L = 2.0    # Length (m)
        self.B = 1.08   # Beam (m)
        self.dof = 2    # Number of DOFs
        self.lx = 1
        self.ly1 = 0.5
        self.ly2 = -0.5
        self.Binv = np.ones((2, 2))

    def step(self, eta: np.ndarray, nu: np.ndarray, prev_u: np.ndarray, action: np.ndarray, beta_c: float, V_c: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Takes in 3-dimensional eta

        Parameters
        ----------
        action : np.ndarray([float, float])
            Control vector
        """

        # Actuation limits
        action[0] = sat(-1, action[0], 1)
        action[1] = sat(D2R(-15), action[1], D2R(15))

        # Kinematics in NED
        v = action[0]*np.cos(eta[-1])
        u = action[0]*np.sin(eta[-1])
        r = action[0]*action[1]

        # Concatenate kinematic vector
        nu_n = np.array([v, u, 0, 0, 0, r])
        nu = N2B(eta).dot(nu_n)

        u_control = action
        return nu, u_control
