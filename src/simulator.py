"""
Simple simulator for driving round boats
github: @rambech

Reference frames
----------------
North-east-down (NED):
    Sometimes denoted {n} is the world reference frame of the 
    vehicle simulation

Body-fixed (BODY):
    Vehicle fixed reference frame, sometimes denoted {b}

Screen:
    The computer screen reference frame, sometimes denoted {s}
"""

import pygame
import numpy as np
from vehicle import Vehicle, Otter
from control import Control, Manual
from maps import SimpleMap, Target
from plotting.data import save_data
from utils import attitudeEuler, B2N, N2B, N2S, N2S2D, D2L, ssa

# Keystroke inputs
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    K_TAB,
    KEYDOWN,
    QUIT,
)

# TODO: Add comments
# TODO: Add function/method descriptions
# TODO: Add rgb_array render mode
# TODO: Add simulator data acquisition
# TODO: Add


class Simulator():
    """
    Simple simulator for testing vehicle dynamics, mostly for the Otter vehicle

    Attributes
    ----------
    vehicle : Vehicle
        Simulated vehicle
    map : Map
        Simulated environment/map
    fps : float
        Simulation refresh rate
    dt : float
        Sample time
    eta_init : np.ndarray
        Initial vehicle pose in NED frame
    eta : np.ndarray
        Current vehicle pose in NED frame
    nu : np.ndarray
        Current vehicle velocity
    u : np.ndarray
        Current control forces
    clock : pygame.time.Clock()
        Simulation clock
    screen : pygame.display
        Simulation screen for visualisation
    quay : SimpleQuay
        Quay the vehicle docks to and interacts with
    target : Target
        Target/desired vehicle pose
    vessel_rect: pygame.Rect
        Vehicle hitbox for environment interaction
    bounds : list[pygame.Rect]
        Environment outer bounds

    Methods
    -------
    simulate()
        Runs the main simulation loop as a pygame instance
    step(tau_d: np.ndarray)
        Calls the vehicle step function and 
        saves the resulting eta, nu and u vectors
    render()
        Renders the vehicle to screen with updated pose
    bump()
        Simulates a fully elastic collision with the quay
    close()
        Closes the display and ends the pygame instance
    """

    def __init__(self, vehicle: Vehicle, control: Control, map: SimpleMap,
                 seed: int = None, target: Target = None,
                 eta_init=np.zeros(6, float), fps=30, data_acq=False) -> None:
        """
        Initialises simulator object

        Parameters
        ----------
        vehicle: Otter
            Simulated vehicle
        map : Map
            Simulated environment/map
        seed : Int
            Seed for random position initialization (default is None)
        target : Target, optional
            Target/desired vehicle pose (default is None)
        eta_init : np.ndarray, optional
            Initial vehicle pose in NED frame (default is np.zeros(6, float))
        fps : float, optional
            Simulation refresh rate (default is 30)


        Returns
        ------
            None
        """

        self.vehicle = vehicle
        self.control = control
        self.map = map
        self.quay = self.map.quay
        self.fps = fps
        self.dt = self.vehicle.dt
        self.seed = seed
        self.eta_d = target.eta_d

        # Initialize data acquisition
        if data_acq == True:
            self.data = {"Control method": self.control.control_type,
                         "Path": [],
                         "u": []}

            if self.control.control_type == "NMPC":
                self.data["eta predictions"] = []
                self.data["u predictions"] = []

        # Simulate vehicle at a higher rate than the RL step
        self.step_rate = 1/(self.dt*self.fps)
        assert (
            self.step_rate % 1 == 0
        ), f"Step rate must be a positive integer, got {self.step_rate}. \
            Make sure the vehicle FPS is a multiple of the simulation FPS"

        # Initial conditions
        if self.seed is not None:
            self.eta = self.random_eta()
        else:
            self.eta_init = eta_init.copy()  # Save initial pose
            self.eta = eta_init              # Initialize pose

        self.nu = np.zeros(6, float)     # Init velocity
        self.u = np.zeros(2, float)      # Init control vector

        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Otter Simulator")
        self.clock = pygame.time.Clock()

        # Make a screen and fill it with a background colour
        self.screen = pygame.display.set_mode(
            [self.map.BOX_WIDTH, self.map.BOX_LENGTH])
        self.screen.fill(self.map.OCEAN_BLUE)

        # Add target
        self.target = target

        # Initialize hitboxes
        self.vessel_rect = self.vehicle.vessel_image.get_rect()
        self.bounds = [(-map.MAP_SIZE[0]/2, -map.MAP_SIZE[1]/2),
                       (map.MAP_SIZE[0]/2, map.MAP_SIZE[1]/2)]
        N_min, E_min, N_max, E_max = self.map.bounds
        self.eta_max = np.array([N_max, E_max, vehicle.limits["psi_max"]])
        self.eta_min = np.array([N_min, E_min, vehicle.limits["psi_min"]])
        self.edges = []
        self.corner = []
        self.closest_edge = ((0, 0), (0, 0))
        self.see_edges = True  # Turn edges and vertices off or on
        self.render()

    def simulate(self):
        """
        Runs the main simulation loop as a pygame instance
        """

        # Run until the user asks to quit or hit something they shouldn't
        running = True
        out_of_bounds = False
        while running:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if self.control.control_type == "Manual":
                        # Manual surge force
                        if event.key == K_UP:
                            # Constant positive surge force
                            X = 5   # [N]

                        elif event.key == K_DOWN:
                            # Constant negative surge force
                            X = -5  # [N]

                        else:
                            X = 0   # [N]

                        # Manual yaw moment
                        if event.key == K_RIGHT:
                            # Constant positive yaw moment
                            N = 5   # [Nm]

                        elif event.key == K_LEFT:
                            # Constant positive yaw moment
                            N = -5  # [Nm]

                        else:
                            N = 0   # [Nm]

                    # Go back to start
                    if event.key == K_TAB:
                        # Go back to initial condition
                        if self.seed is not None:
                            self.eta = self.random_eta()
                        else:
                            # Note: use copy() when copying arrays
                            self.eta = self.eta_init.copy()

                        self.nu = np.zeros(6, float)
                        self.u = np.zeros(2, float)

                    # Quit if escape key is pressed
                    if event.key == K_ESCAPE:
                        running = False

                else:
                    X = 0   # [N]
                    N = 0   # [Nm]

                # Quit if window is closed by user
                if event.type == QUIT:
                    running = False

            if self.vehicle != None:
                # Manual force input
                tau_d = np.array([X, N])

                # obs = self.get_observation()

                # print(f"Observation: \n \
                # delta_x:    {obs[0]} \n \
                # delta_y:    {obs[1]} \n \
                # delta_psi:  {obs[2]} \n \
                # u:          {obs[3]} \n \
                # v:          {obs[4]} \n \
                # r:          {obs[5]} \n \
                # d_q:        {obs[6]} \n \
                # psi_q:      {R2D(obs[7])} \n \
                # d_o:        {obs[8]} \n \
                # psi_o:      {R2D(obs[9])} \n")

                if self.crashed():
                    running = False

                # Step vehicle simulation
                if not out_of_bounds:
                    if self.control.control_type == "Manual":
                        self.manual_step(tau_d)
                    else:
                        self.step()

            self.render()
        self.close()

    def step(self):
        """
        Calls the vehicle step function and 
        saves the resulting eta, nu and u vectors

        Parameters
        ----------
        self
        """
        # print(f"self.eta.shape: {self.eta.shape}")
        # print(f"self.eta[:2].shape: {self.eta[:2].shape}")
        # print(f"self.eta[-1].shape: {self.eta[-1:].shape}")

        # Control step
        x_init = np.concatenate([self.eta[:2], self.eta[-1:],
                                 self.nu[:2], self.nu[-1:]])
        x, u_control = self.control.step(x_init, self.u, self.eta_d)

        u_control = u_control[:, 1]

        # print(f"predicted x: {x}")
        # print(f"u_control: {u_control}")

        # Dynamic step
        for _ in range(int(self.step_rate)):
            # Kinetic step
            self.nu, self.u = self.vehicle.step(
                self.eta, self.nu, self.u, u_control, self.map.SIDESLIP, self.map.CURRENT_MAGNITUDE)
            # TODO: Change sideslip and current magnitude source

            # print(f"self.nu: {self.nu}")
            # print(f"self.u: {self.u}")

            # Kinematic step
            self.eta = attitudeEuler(self.eta, self.nu, self.dt)

        self.corner = self.vehicle.corners(self.eta)

    def manual_step(self, tau_d: np.ndarray):
        """
        Calls the vehicle step function and 
        saves the resulting eta, nu and u vectors

        Parameters
        ----------
        tau_d: np.ndarray
            Vector of desired surge force X and yaw momentum N
        """

        # Control step
        u_control = self.vehicle.unconstrained_allocation(tau_d)
        u_control = self.vehicle._normalise(u_control)

        # print(f"u_control: {u_control}")

        # Dynamic step
        for _ in range(int(self.step_rate)):
            # Kinetic step
            self.nu, self.u = self.vehicle.step(
                self.eta, self.nu, self.u, u_control, self.map.SIDESLIP, self.map.CURRENT_MAGNITUDE)
            # TODO: Change sideslip and current magnitude source

            # Kinematic step
            self.eta = attitudeEuler(self.eta, self.nu, self.dt)

        self.corner = self.vehicle.corners(self.eta)

    def render(self):
        """
        Updates screen with the changes that has happened with
        the vehicle and map/environment
        """

        self.screen.fill(self.map.OCEAN_BLUE)
        edges = []

        # Add outer bounds of map
        for obstacle in self.map.obstacles:
            self.screen.blit(obstacle.surf, obstacle.rect)
            edges.append(obstacle.colliding_edge)

        self.edges = edges

        # Render target pose to screen
        if self.target != None:
            self.screen.blit(self.target.image, self.target.rect)

        # Render quay to screen
        self.screen.blit(self.quay.surf, self.quay.rect)

        # if True:
        #     for i in range(1000):
        #         x, y = N2S(self.random_eta(),
        #                    self.map.scale, self.map.origin)[0:2].tolist()

        #         pygame.draw.circle(self.screen, (255, 26, 117),
        #                            (x, y), 2)

        # Render vehicle to screen
        if self.vehicle != None:
            vessel_image, self.vessel_rect = self.vehicle.render(
                self.eta, self.map.origin)
            # print(f"origin: {self.map.origin}")
            # print(f"eta_n: {self.eta}")

            self.screen.blit(vessel_image, self.vessel_rect)

            # Speedometer
            U = np.linalg.norm(self.nu[0:2], 2)
            font = pygame.font.SysFont("Times New Roman", 12)
            speed = font.render(f"SOG: {np.round(U, 2)} [m/s]", 1, (0, 0, 0))
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

            if self.see_edges:
                for corner in self.corner:
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

        pygame.display.flip()
        self.clock.tick(self.fps)

    def crashed(self) -> bool:
        for corner in self.vehicle.corners(self.eta):
            _, dist_corner_quay = D2L(self.quay.colliding_edge, corner)
            _, dist_corner_obs = D2L(self.closest_edge, corner)
            if dist_corner_obs < 0.01:  # If vessel touches obstacle, simulation stops
                return True
            elif abs(corner[0]) >= self.eta_max[0] or abs(corner[1]) >= self.eta_max[1]:
                return True
            elif dist_corner_quay < 0.05:
                self.bump()

    def bump(self):
        """
        Simulates a fully elastic collision between the quay and the vessel
        """

        # Transform nu from {b} to {n}
        nu_n = B2N(self.eta).dot(self.nu)

        # Send the vessel back with the same speed it came in
        U_n = np.linalg.norm(nu_n[0:3], 3)
        min_U_n = -U_n

        # Necessary angles
        beta = np.arctan(nu_n[2]/nu_n[0])   # Sideslip
        alpha = np.arcsin(nu_n[1]/min_U_n)  # Angle of attack

        nu_n[0:3] = np.array([min_U_n*np.cos(alpha)*np.cos(beta),
                              min_U_n*np.sin(beta),
                              min_U_n*np.sin(alpha)*np.cos(beta)])
        self.nu = N2B(self.eta).dot(nu_n)

    def out_of_bounds(self, vertex):
        return vertex[0] <= self.bounds[0][0] or vertex[1] <= self.bounds[0][1] or \
            vertex[0] >= self.bounds[1][0] or vertex[1] >= self.bounds[1][1]

    def random_eta(self):
        padding = 2  # [m]
        x_init = np.random.uniform(
            self.map.bounds[0] + padding, self.map.bounds[2] - self.quay.length - padding)
        y_init = np.random.uniform(
            self.map.bounds[1] + padding, self.map.bounds[3] - padding)
        psi_init = ssa(np.random.uniform(-np.pi, np.pi))

        return np.array([x_init, y_init, 0, 0, 0, psi_init], float)

    def get_observation(self):
        delta_eta = self.eta - self.eta_d
        delta_eta_2D = np.concatenate(
            (delta_eta[0:2], delta_eta[-1]), axis=None)
        d_q, psi_q = self.direction_and_angle_to_quay()
        d_o, psi_o = self.direction_and_angle_to_obs()

        return np.concatenate((delta_eta_2D, self.nu[0:3], d_q, psi_q, d_o, psi_o),
                              axis=None).astype(np.float32)

    def direction_and_angle_to_obs(self):
        angle = 0
        dist = np.inf
        for edge in self.edges:
            bearing, range = D2L(edge, self.eta[0:2])
            if range < dist:
                angle = bearing - self.eta[-1]
                dist = range
                self.closest_edge = edge

        return dist, angle

    def direction_and_angle_to_quay(self):
        bearing, dist = D2L(self.quay.colliding_edge, self.eta[0:2])
        angle = bearing - self.eta[-1]

        return dist, angle

    def close(self):
        if self.data:
            save_data(self.data, type(self).__name__)

        pygame.display.quit()
        pygame.quit()


def test_simulator():
    """
    Procedure for testing simulator
    """
    # Initialize constants
    fps = 20
    eta_init = np.array([0, 0, 0, 0, 0, 0], float)
    eta_d = np.array([25/2-0.75-1, 0, 0, 0, 0, 0], float)

    # Initialize vehicle
    vehicle = Otter(dt=1/fps)
    control = Manual()

    map = SimpleMap()
    target = Target(eta_d, vehicle, map.origin)
    simulator = Simulator(vehicle, control, map, None, target,
                          eta_init=eta_init, fps=fps)
    simulator.simulate()


# test_simulator()
