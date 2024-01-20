"""
This is the base class for all my environments
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

from vehicle import Otter
from maps import SimpleMap, Target
from utils import attitudeEuler, D2L, D2R, ssa, R2D, N2S, N2S2D


class Env(gym.Env):
    """
    Base class for gym environments
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 20
    }

    def __init__(self) -> None:
        super(Env, self).__init__()
        """
        Initialises DPEnv() object
        """

        # Must be overwritten
        self.vehicle = None
        self.dt = None
        self.bounds = None
        self.fps = self.metadata["render_fps"]

        self.seed = None

        # ---
        # Map
        # ---
        self.map = None
        self.eta_d = None
        self.target = None
        self.obstacles = None
        self.quay = None
        self.closest_edge = ((0, 0), (0, 0))

        # Initial conditions
        self.eta = np.zeros(6, float)       # Init eta
        self.eta_init = self.eta.copy()     # Save init eta
        self.nu = np.zeros(6, float)        # Init velocity
        self.u = np.zeros(2, float)         # Init control vector

        # Weather
        self.random_weather = None
        self.beta_c, self.V_c = self.current_force()

        # -----------------
        # Observation space
        # -----------------
        self.observation_space = None

        # ------------
        # Action space
        # ------------
        self.action_space = None

        # --------------
        # End conditions
        # --------------
        self.eta_max = None

        # Fail
        seconds_limit = 60
        self.step_limit = seconds_limit*self.fps  # [step]
        self.step_count = 0

        # Success
        # Must be overwritten
        self.thres = None             # [m, rad]
        self.stay_time = None  # [step]
        self.stay_timer = None

        self.prev_shape = None
        self.prev_obs = None

    def reset(self, seed=None):
        if self.seed is not None:
            self.eta = self.random_eta()
        else:
            self.eta = self.eta_init.copy()

        self.beta_c, self.V_c = self.current_force()

        self.nu = np.zeros(6, float)
        self.u = np.zeros(2, float)

        # self.has_crashed = False
        self.stay_timer = None
        self.step_count = 0
        self.prev_dist = None
        self.prev_obs = None
        self.touched_quay = False

        observation = self.get_observation()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self):
        ...

    def render(self):
        """
        Updates screen with the changes that has happened with
        the vehicle and map/environment
        """

        if not self.screen and self.render_mode == "human":
            # Initialize pygame
            pygame.init()
            pygame.display.set_caption("Otter RL")
            self.clock = pygame.time.Clock()

            # Make a screen and fill it with a background colour
            self.screen = pygame.display.set_mode(
                [self.map.BOX_WIDTH, self.map.BOX_LENGTH])

        self.screen.fill(self.map.OCEAN_BLUE)

        for obstacle in self.obstacles:
            self.screen.blit(obstacle.surf, obstacle.rect)

        self.screen.blit(self.quay.surf, self.quay.rect)

        # if True:
        #     for i in range(1000):
        #         x, y = N2S(self.random_eta(),
        #                    self.map.scale, self.map.origin)[0:2].tolist()

        #         pygame.draw.circle(self.screen, (255, 26, 117),
        #                            (x, y), 2)

        # Render target pose to screen
        self.screen.blit(self.target.image, self.target.rect)

        # Render vehicle to screen
        vessel_image, self.vessel_rect = self.vehicle.render(
            self.eta, self.map.origin)
        self.screen.blit(vessel_image, self.vessel_rect)

        # Speedometer
        U = np.linalg.norm(self.nu[0:2], 2)
        font = pygame.font.SysFont("Times New Roman", 12)
        speed = font.render(
            f"SOG: {np.round(U, 2)} [m/s]", 1, (0, 0, 0))
        self.screen.blit(speed, (10, self.map.BOX_LENGTH-20))

        # Position
        x = np.round(self.eta[0])
        y = np.round(self.eta[1])
        position = font.render(f"NED: ({x}, {y})", 1, (0, 0, 0))
        self.screen.blit(position, (10, self.map.BOX_LENGTH-32))

        # Thruster revolutions
        n1 = np.round(self.u[0])
        n2 = np.round(self.u[1])
        rpm = font.render(f"THR: ({n1}, {n2} [%])", 1, (0, 0, 0))
        self.screen.blit(rpm, (10, self.map.BOX_LENGTH-44))

        # Weather
        beta_c = np.round(self.beta_c, 2)
        V_c = np.round(self.V_c, 2)
        current = font.render(
            f"WTR: ({V_c}, {R2D(beta_c)} [m/s, degrees])", 1, (0, 0, 0))
        self.screen.blit(current, (10, self.map.BOX_LENGTH-56))

        if self.quay and True:
            for corner in self.vehicle.corners(self.eta):
                corner_n = np.array([corner[0], corner[1], 0, 0, 0, 0])
                corner_s = N2S(corner_n, self.vehicle.scale,
                               self.map.origin)
                pygame.draw.circle(self.screen, (255, 26, 117),
                                   (corner_s[0], corner_s[1]), 2)

            pygame.draw.line(self.screen, (136, 77, 255),
                             N2S2D(self.quay.colliding_edge[0], self.map.scale, self.map.origin), N2S2D(self.quay.colliding_edge[1], self.map.scale, self.map.origin), 2)

            for edge in self.edges:
                pygame.draw.line(self.screen, (255, 26, 117),
                                 N2S2D(edge[0], self.map.scale, self.map.origin), N2S2D(edge[1], self.map.scale, self.map.origin), 2)

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.screen:
            pygame.display.quit()
            pygame.quit()

    def get_observation(self):
        ...

    def crashed(self) -> bool:
        ...

    def current_force(self) -> tuple[float, float]:
        """
        Three modes, random changing, keystroke and random static

        Random changing
        ---------------
        The current can change in both magnitude and direction during
        an episode

        Keystroke
        ---------
        Current changes during an episode by user input via the arrow keys

        Random static
        -------------
        The current is random in magnitude and direction, but is static
        through an episode

        Parameters
        ----------
            self

        Returns
        -------
            beta_c, V_c : tuple[float, float]
                Angle and magnitude of current
        """
        if self.random_weather:
            beta_c = ssa(np.random.uniform(0, 2*np.pi))
            V_c = np.random.uniform(0, 1.03)
        else:
            beta_c = 0.0
            V_c = 0.0

        return beta_c, V_c

    def random_eta(self):
        ...

    def in_area(self):
        dist = np.linalg.norm(self.eta[0:2] - self.eta_d[0:2], 2)
        ang = abs(self.eta[-1] - self.eta_d[-1])
        if dist <= self.thres[0] and ang <= self.thres[1]/2:
            return True

        return False

    def success(self):
        if self.stay_timer is not None:
            if int(self.stay_timer) >= self.stay_time:
                return True

        return False
