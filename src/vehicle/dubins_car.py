import numpy as np
from .vehicle import Vehicle
import pygame


class DubinsCar(Vehicle):
    """
    Simplest possible vehicle model, based on Dubins Car
    with added controllable velocity
    https://lavalle.pl/planning/node821.html
    """

    def __init__(self, seed=None, dt=0.02) -> None:
        super().__init__(seed, dt)
        self.vessel_image = pygame.image.load(
            'vehicle/images/otter.png')
        self.vessel_image = pygame.transform.scale(
            self.vessel_image, (self.scale*self.B, self.scale*self.L))

    def _init_model(self):
        # Constants
        self.g = 9.81   # acceleration of gravity (m/s^2)

        # Initialize the Otter USV model
        self.T_n = 1.0  # Propeller time constants (s)
        self.L = 2.0    # Length (m)
        self.B = 1.08   # Beam (m)
        self.dof = 2    # Number of DOFs

    def dynamics_step(self, eta: np.ndarray, nu: np.ndarray, prev_u: np.ndarray, action: np.ndarray, beta_c: float, V_c: float) -> (np.ndarray, np.ndarray):
        """
        Takes in 3-dimensional eta

        Parameters
        ----------
        action : np.ndarray([float, float])
            Control vector
        """

        # Kinematics
        v = action[0]*np.cos(eta[-1])
        u = action[0]*np.sin(eta[-1])
        r = action[0]*action[1]

        # Concatenate kinematic vector
        nu = np.array([v, u, r])

        u_control = action
        return nu, u_control
