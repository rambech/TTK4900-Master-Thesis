"""
This is the simplest python RL environment I could think of.
With a large boundary around this is simply snake with one pixel


Observations space
------------------
s = [x_error, y_error, psi_error, u, v, r]


Action space
------------
a = [n_1, n_2]

"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

from vehicle import Otter
from maps import SimpleMap, Target
from utils import attitudeEuler, D2L, D2R, ssa, R2D

from rl.rewards import r_gaussian, r_time, r_surge, r_euclidean, r_in_area

# Environment parameters
FPS = 50                          # [fps] Frames per second

# I have chosen the origin of NED positon to be
# in the middle of the screen. This means that
# the pygame coordinates are different to the
# NED ones, RL and everything else is calculated
# in NED, only rendering happens in the other coordinates.


class DPEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": FPS,
    }

    def __init__(self, vehicle: Otter, map: SimpleMap, seed: int = None, render_mode=None, FPS: int = 50, eta_init=np.zeros(6, float),  threshold=1, random_weather=False) -> None:
        super(DPEnv, self).__init__()
        """
        Initialises DPEnv() object
        """

        self.vehicle = vehicle
        self.map = map
        self.eta_d = np.zeros(6, float)
        self.target = Target(self.eta_d, vehicle, map.origin)
        self.dt = self.vehicle.dt
        self.bounds = self.map.bounds
        self.fps = FPS

        self.seed = seed

        # Initial conditions
        if seed is not None:
            # Make random initial condition and weather
            np.random.seed(seed)
            self.eta = self.random_eta()

        else:
            # Use default initial conditions and weather
            self.eta_init = eta_init.copy()  # Save initial pose
            self.eta = eta_init              # Initialize pose
            self.nu = np.zeros(6, float)     # Init velocity
            self.u = np.zeros(2, float)      # Init control vector

        # Weather
        self.random_weather = random_weather
        self.beta_c, self.V_c = self.current_force()

        # -----------------
        # Observation space
        # -----------------
        N_min, E_min, N_max, E_max = self.bounds
        self.eta_max = np.array([N_max, E_max, vehicle.limits["psi_max"]])
        self.eta_min = np.array([N_min, E_min, vehicle.limits["psi_min"]])
        self.nu_max = vehicle.limits["nu_max"]
        self.nu_min = vehicle.limits["nu_min"]

        upper = np.concatenate(
            (self.eta_max, self.nu_max), axis=None).astype(np.float32)
        lower = np.concatenate(
            (self.eta_min, self.nu_min), axis=None).astype(np.float32)

        self.observation_size = (upper.size,)

        self.observation_space = spaces.Box(
            lower, upper, self.observation_size)
        self.action_space = vehicle.action_space

        # --------------
        # End conditions
        # --------------
        # Fail
        seconds_limit = 120
        self.step_limit = seconds_limit*FPS  # [step]
        self.step_count = 0

        # Success
        s_seconds = 5
        self.thres = threshold          # [m, rad]
        self.stay_time = FPS*s_seconds  # [step]
        self.stay_timer = None

        self.prev_shape = None

        # ---------
        # Rendering
        # ---------
        self.metadata["render_fps"] = FPS

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            # Initialize pygame
            pygame.init()
            pygame.display.set_caption("Otter RL")
            self.clock = pygame.time.Clock()

            # Make a screen and fill it with a background colour
            self.screen = pygame.display.set_mode(
                [self.map.BOX_WIDTH, self.map.BOX_LENGTH])
            self.screen.fill(self.map.OCEAN_BLUE)
            # self.particles = []

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

        self.prev_shape = None

        observation = self.get_observation()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        terminated = False
        # self.step_count += 1

        # Simulate vehicle at a higher rate than the RL step
        step_rate = 1/(self.dt*self.fps)
        assert (
            step_rate % 1 == 0
        ), f"Step rate must be a positive integer, got {step_rate}. \
            Make sure the vehicle FPS is a multiple of the RL FPS"

        for _ in range(int(step_rate)):
            self.nu, self.u = self.vehicle.step(
                self.eta, self.nu, self.u, action, self.beta_c, self.V_c)
            self.eta = attitudeEuler(self.eta, self.nu, self.dt)

        observation = self.get_observation()

        shape = r_euclidean(observation)
        reward = 0
        if self.prev_shape:
            reward = shape - self.prev_shape

        reward += r_in_area(self.in_area())
        reward += r_time()

        self.prev_shape = shape

        if self.in_area():
            if self.stay_timer is None:
                self.stay_timer = 0
            else:
                self.stay_timer += 1

            # Give reward if inside area
            # reward += 1
        else:
            self.stay_timer = None

        if self.success():
            terminated = True
            reward = 1000

        if self.crashed():
            terminated = True
            reward = -1000

        # if self.step_count >= self.step_limit:
        #     terminated = True
        #     reward = -100

        if self.render_mode == "human":
            self.render()

        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

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

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.screen:
            pygame.display.quit()
            pygame.quit()

    def get_observation(self):
        eta_error = self.eta - self.eta_d
        pos_error = np.concatenate(
            (eta_error[0:2], ssa(eta_error[-1])), axis=None)
        vel_error = np.concatenate(
            (self.nu[0:2], self.nu[-1]), axis=None)

        return np.concatenate((pos_error, vel_error),
                              axis=None).astype(np.float32)

    def crashed(self) -> bool:
        for corner in self.vehicle.corners(self.eta):
            if abs(corner[0]) >= self.eta_max[0] or abs(corner[1]) >= self.eta_max[1]:
                return True

        return False

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
        """
        Spawn vehicle based on uniform distribution. 
        2 meter buffer at the edges 

        Parameters
        ----------
        self

        Returns
        -------
        eta_init : np.ndarray
            Random initial position

        """
        padding = 2  # [m]
        x_init = np.random.uniform(
            self.bounds[0] + padding, self.bounds[2] - padding)
        y_init = np.random.uniform(
            self.bounds[1] + padding, self.bounds[3] - padding)
        ang2d = np.arctan2(
            y_init - self.eta_d[1], x_init - self.eta_d[0],) - np.pi
        psi_init = np.random.uniform(ang2d-np.pi/2, ang2d+np.pi/2)

        return np.array([x_init, y_init, 0, 0, 0, psi_init], float)

    def in_area(self):
        dist = np.linalg.norm(self.eta[0:2] - self.eta_d[0:2], 2)
        if dist <= self.thres:
            return True

        return False

    def success(self):
        if self.stay_timer is not None:
            if int(self.stay_timer) >= self.stay_time:
                return True

        return False
