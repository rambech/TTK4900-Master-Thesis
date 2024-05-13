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
import time
from vehicle import Vehicle, Otter
from control import Control, Manual
from maps import SimpleMap, Target
from plotting.data import save_data
import utils

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
# TODO: Don't initialize rendering unless render=True


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
                 eta_init=np.zeros(6, float), fps=30, data_acq=False, render=True) -> None:
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
        self.stay_timer = 0
        self.stay_time = 2
        self.threshold = 1
        self.heading_threshold = utils.D2R(15)

        self.bool_render = render
        self.error_caught = False

        # Initialize data acquisition
        if data_acq == True:
            self.data = {"Control method": self.control.control_type,
                         "Config": self.control.config,
                         "target": self.eta_d.tolist(),
                         "Path": [],
                         "u": [],
                         "slack": []}

            if self.control.control_type == "NMPC":
                self.data["total time"] = 0.0
                self.data["average time"] = 0.0
                self.data["num control intervals"] = 0
                self.data["time max"] = 0
                self.data["time min"] = 0
                self.data["time std deviation"] = 0
                self.data["time"] = []
                self.data["state predictions"] = []
                self.data["control predictions"] = []

        # Simulate vehicle at a higher rate than the RL step
        self.step_rate = 1/(self.dt*self.fps)
        assert (
            self.step_rate % 1 == 0
        ), f"Step rate must be a positive integer, got {self.step_rate}. \
            Make sure the vehicle FPS is a multiple of the simulation FPS"

        self.count = 0

        # Initial conditions
        if self.seed is not None:
            self.eta = self.random_eta()
        else:
            self.eta_init = eta_init.copy()  # Save initial pose
            self.eta = eta_init              # Initialize pose

        self.nu = 0.001*np.ones(6, float)     # Init velocity
        self.u = 0.001*np.ones(2, float)      # Init control vector
        self.x_pred = np.concatenate([self.eta, self.nu])

        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Otter Simulator")
        self.clock = pygame.time.Clock()

        if self.bool_render:
            # Make a screen and fill it with a background colour
            self.screen = pygame.display.set_mode(
                [self.map.BOX_WIDTH, self.map.BOX_HEIGHT])
            self.screen.fill(self.map.OCEAN_BLUE)

            # Add target
            self.target = target

            # Initialize hitboxes
            self.vehicle.init_render(self.map.scale)
            self.vessel_rect = self.vehicle.vessel_image.get_rect()
        self.bounds = [(-map.MAP_SIZE[0]/2, -map.MAP_SIZE[1]/2),
                       (map.MAP_SIZE[0]/2, map.MAP_SIZE[1]/2)]
        N_min, E_min, N_max, E_max = self.map.bounds
        self.eta_max = np.array([N_max, E_max, vehicle.limits["psi_max"]])
        self.eta_min = np.array([N_min, E_min, vehicle.limits["psi_min"]])
        self.edges = []
        self.corner = []
        self.closest_edge = ((0, 0), (0, 0))

        if self.bool_render:
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

                # Step vehicle simulation
                if not out_of_bounds:
                    if self.control.control_type == "Manual":
                        self.manual_step(tau_d)
                    else:
                        self.step()

                if self.docked():
                    self.stay_timer += 1/self.fps

                if self.crashed():
                    print("Crashed :((")
                    running = False
                elif self.success():
                    print("Success :))")
                    running = False

                elif self.error_caught:
                    print("Stopping")
                    running = False

            if self.bool_render:
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

        # Control step
        x_init = np.concatenate([self.eta[:2], self.eta[-1:],
                                 self.nu[:2], self.nu[-1:]])

        print("===================================")
        print("------------- Running -------------")
        print(f"Actual t{self.count}:")
        print(
            f"Vessel pose:  ({np.round(x_init[0],4)}, {np.round(x_init[1],4)}, {np.round(x_init[2],4)})")
        print(
            f"Target pose:  ({np.round(self.eta_d[0],4)}, {np.round(self.eta_d[1],4)}, {np.round(self.eta_d[2],4)})")
        print(
            f"Vel:          ({np.round(x_init[3],4)}, {np.round(x_init[4],4)}, {np.round(x_init[5],4)})")
        print(
            f"Thrust:       ({np.round(self.u[0],4)}, {np.round(self.u[1],4)})")

        t0 = time.time()    # Start time

        # x, u_control = self.control.step(x_init, self.u, self.eta_d)

        # try:
        #     x, u_control = self.control.step(x_init, self.u, self.eta_d)
        # except RuntimeError as error:
        #     x = None
        #     u_control = np.zeros(2)
        #     self.error_caught = True
        #     print("Error caught", error)

        x, u_control, slack, self.error_caught = self.control.debug(x_init,
                                                                    self.u,
                                                                    self.eta_d)

        t1 = time.time()    # End time

        t = t1 - t0

        try:
            # TODO: Fix data acquisition
            self.data["time"].append(t)
            if x is not None:
                self.data["state predictions"].append(x.tolist())
                self.data["control predictions"].append(u_control.tolist())
                self.data["slack"].append(slack.tolist())
            small_eta = np.array([self.eta[0], self.eta[1], self.eta[-1]])
            self.data["Path"].append(small_eta.tolist())
            self.data["u"].append(self.u.tolist())
        except AttributeError:
            print("Could not collect data point")

        if x is not None:
            u_control = u_control[:, 1]
            self.u_pred = u_control
            self.x_pred = x[:, 1]

            print(f"Predicted t{self.count+1}:")
            print(
                f"Pose:             ({np.round(x[0, 1],4)}, {np.round(x[1, 1],4)}, {np.round(x[2, 1],4)})")
            print(
                f"Pred final pose:  ({np.round(x[0, -1],4)}, {np.round(x[1, -1],4)}, {np.round(x[2, -1],4)})")
            print(
                f"Vel:              ({np.round(x[3, 1],4)}, {np.round(x[4, 1],4)}, {np.round(x[5, 1],4)})")
            print(
                f"Thrust:           ({np.round(u_control[0],4)}, {np.round(u_control[1],4)})")
        print("===================================")

        # Dynamic step
        for _ in range(int(self.step_rate)):
            # Kinetic step
            self.nu, self.u = self.vehicle.step(
                self.eta, self.nu, self.u, u_control, self.map.SIDESLIP, self.map.CURRENT_MAGNITUDE)
            # TODO: Change sideslip and current magnitude source

            # print(f"self.nu: {self.nu}")
            # print(f"self.u: {self.u}")

            # Kinematic step
            self.eta = utils.attitudeEuler(self.eta, self.nu, self.dt)

            # print(f"Loop eta: {self.eta}")

        self.corner = self.vehicle.corners(self.eta)

        self.count += 1

        # Stop if speed and thrust are zero:
        if (self.count > 10 and np.round(self.u[0], 4) == 0.0 and
                np.round(self.u[0], 4) == 0.0 and np.round(self.eta[3], 4) == 0.0):
            self.error_caught = True

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
        # u_control = self.vehicle._normalise(u_control)

        # print(f"u_control: {u_control}")

        # Dynamic step
        for _ in range(int(self.step_rate)):
            # Kinetic step
            self.nu, self.u = self.vehicle.step(
                self.eta, self.nu, self.u, u_control, self.map.SIDESLIP, self.map.CURRENT_MAGNITUDE)
            # TODO: Change sideslip and current magnitude source

            # Kinematic step
            self.eta = utils.attitudeEuler(self.eta, self.nu, self.dt)

            if self.crashed:
                break

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

        if False:
            for i in range(1000):
                x, y = N2S(self.random_eta(),
                           self.map.scale, self.map.origin)[0:2].tolist()

                pygame.draw.circle(self.screen, (255, 26, 117),
                                   (x, y), 2)

        # Render vehicle to screen
        if self.vehicle != None:
            self.show_pred(self.data["state predictions"])
            self.show_path(self.data["Path"])
            self.show_harbour()

            vessel_image, self.vessel_rect = self.vehicle.render(
                self.eta, self.map.origin)
            # print(f"origin: {self.map.origin}")
            # print(f"eta_n: {self.eta}")

            self.screen.blit(vessel_image, self.vessel_rect)

            # Speedometer
            U = np.linalg.norm(self.nu[0:2], 2)
            font = pygame.font.SysFont("Times New Roman", 12)
            speed = font.render(f"SOG: {np.round(U, 2)} [m/s]", 1, (0, 0, 0))
            self.screen.blit(speed, (10, self.map.BOX_HEIGHT-20))

            # Position
            x = np.round(self.eta[0])
            y = np.round(self.eta[1])
            position = font.render(f"NED: ({x}, {y})", 1, (0, 0, 0))
            self.screen.blit(position, (10, self.map.BOX_HEIGHT-32))

            # Thruster revolutions
            n1 = np.round(self.u[0])
            n2 = np.round(self.u[1])
            rpm = font.render(f"THR: ({n1}, {n2})[%]", 1, (0, 0, 0))
            self.screen.blit(rpm, (10, self.map.BOX_HEIGHT-44))

            # Visualise safety bounds
            # buffer = 0.2    # meters
            buffer = 0.0
            half_length = self.vehicle.L/2 + buffer
            half_beam = self.vehicle.B/2 + buffer
            safety_bounds = np.array([[half_length, half_beam],
                                      [half_length, -half_beam],
                                      [-half_length, -half_beam],
                                      [-half_length, half_beam]])

            eta_bounds = []
            for bound in safety_bounds:
                eta_bounds.append(utils.R(self.eta[-1]) @ bound + self.eta[:2])

            # print(f"eta_bounds: {eta_bounds}")
            # print(f"corners:    {self.corner}")

            pygame.draw.line(self.screen, (62, 98, 138),
                             utils.N2S2D(
                                 eta_bounds[0], self.map.scale, self.map.origin),
                             utils.N2S2D(eta_bounds[1], self.map.scale, self.map.origin), 2)
            pygame.draw.line(self.screen, (62, 98, 138),
                             utils.N2S2D(
                                 eta_bounds[1], self.map.scale, self.map.origin),
                             utils.N2S2D(eta_bounds[2], self.map.scale, self.map.origin), 2)
            pygame.draw.line(self.screen, (62, 98, 138),
                             utils.N2S2D(
                                 eta_bounds[2], self.map.scale, self.map.origin),
                             utils.N2S2D(eta_bounds[3], self.map.scale, self.map.origin), 2)
            pygame.draw.line(self.screen, (62, 98, 138),
                             utils.N2S2D(
                                 eta_bounds[3], self.map.scale, self.map.origin),
                             utils.N2S2D(eta_bounds[0], self.map.scale, self.map.origin), 2)

            if self.see_edges:
                for corner in self.corner:
                    corner_n = np.array([corner[0], corner[1], 0, 0, 0, 0])
                    corner_s = utils.N2S(corner_n, self.vehicle.scale,
                                         self.map.origin)
                    pygame.draw.circle(self.screen, (255, 26, 117),
                                       (corner_s[0], corner_s[1]), 2)

                pygame.draw.line(self.screen, (136, 77, 255),
                                 utils.N2S2D(self.quay.colliding_edge[0], self.map.scale, self.map.origin), utils.N2S2D(self.quay.colliding_edge[1], self.map.scale, self.map.origin), 2)

                for edge in self.edges:
                    pygame.draw.line(self.screen, (255, 26, 117),
                                     utils.N2S2D(edge[0], self.map.scale, self.map.origin), utils.N2S2D(edge[1], self.map.scale, self.map.origin), 2)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def show_pred(self, x_pred):
        if x_pred == []:
            return

        last_pred = x_pred[-1]
        for dot in zip(last_pred[0], last_pred[1]):
            point = utils.N2S2D(dot, self.map.scale, self.map.origin)
            pygame.draw.circle(self.screen, (51, 204, 51), point, 1)

    def show_path(self, path):
        if path == []:
            return

        for dot in path:
            # print(f"dot {dot}")
            point = utils.N2S2D(dot[:2], self.map.scale, self.map.origin)
            # print(f"point {point}")
            pygame.draw.circle(self.screen, (244, 172, 103), point, 1)

    def show_harbour(self):
        for i in range(1, len(self.map.convex_set)):
            p1 = utils.N2S2D(self.map.convex_set[i-1],
                             self.map.scale, self.map.origin)
            p2 = utils.N2S2D(self.map.convex_set[i],
                             self.map.scale, self.map.origin)
            pygame.draw.line(self.screen, (255, 0, 0), p1, p2, 2)

        p1 = utils.N2S2D(self.map.convex_set[-2],
                         self.map.scale, self.map.origin)
        p2 = utils.N2S2D(self.map.convex_set[-1],
                         self.map.scale, self.map.origin)
        pygame.draw.line(self.screen, (255, 0, 0), p1, p2, 2)

        p_first = utils.N2S2D(self.map.convex_set[0],
                              self.map.scale, self.map.origin)
        p_last = utils.N2S2D(self.map.convex_set[-1],
                             self.map.scale, self.map.origin)
        # Close loop
        pygame.draw.line(self.screen, (255, 0, 0), p_first, p_last, 2)

    def crashed(self) -> bool:
        for corner in self.vehicle.corners(self.eta):
            _, dist_corner_quay = utils.D2L(self.quay.colliding_edge, corner)
            _, dist_corner_obs = utils.D2L(self.closest_edge, corner)
            if dist_corner_obs < 0.01:  # If vessel touches obstacle, simulation stops
                return True
            elif abs(corner[0]) >= self.eta_max[0] or abs(corner[1]) >= self.eta_max[1]:
                return True
            elif dist_corner_quay < 0.05:
                self.bump()

    def success(self) -> bool:
        if self.stay_timer is not None:
            if int(self.stay_timer) >= self.stay_time:
                return True

        return False

    def docked(self) -> bool:
        if (np.linalg.norm(self.eta[:2] - self.eta_d[:2]) < self.threshold and
                abs(utils.ssa(self.eta[-1] - self.eta_d[-1])) < self.heading_threshold):
            print(f"Desired: {self.eta_d[-1]}")
            print(f"Current: {self.eta[-1]}")
            print(f"True angle error: {self.eta[-1] - self.eta_d[-1]}")
            print(
                f"Smallest angle error: {abs(utils.ssa(self.eta[2] - self.eta_d[2]))}")
            return True

        return False

    def bump(self):
        """
        Simulates a fully elastic collision between the quay and the vessel
        """

        # Transform nu from {b} to {n}
        nu_n = utils.B2N(self.eta).dot(self.nu)

        # Send the vessel back with the same speed it came in
        U_n = np.linalg.norm(nu_n[0:3], 3)
        min_U_n = -U_n

        # Necessary angles
        beta = np.arctan(nu_n[2]/nu_n[0])   # Sideslip
        alpha = np.arcsin(nu_n[1]/min_U_n)  # Angle of attack

        nu_n[0:3] = np.array([min_U_n*np.cos(alpha)*np.cos(beta),
                              min_U_n*np.sin(beta),
                              min_U_n*np.sin(alpha)*np.cos(beta)])
        self.nu = utils.N2B(self.eta).dot(nu_n)

    def out_of_bounds(self, vertex):
        return vertex[0] <= self.bounds[0][0] or vertex[1] <= self.bounds[0][1] or \
            vertex[0] >= self.bounds[1][0] or vertex[1] >= self.bounds[1][1]

    def random_eta(self):
        padding = 2  # [m]
        x_init = np.random.uniform(
            self.map.bounds[0] + padding, self.map.bounds[2] - self.quay.length - padding)
        y_init = np.random.uniform(
            self.map.bounds[1] + padding, self.map.bounds[3] - padding)
        psi_init = utils.ssa(np.random.uniform(-np.pi, np.pi))

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
            bearing, range = utils.D2L(edge, self.eta[0:2])
            if range < dist:
                angle = bearing - self.eta[-1]
                dist = range
                self.closest_edge = edge

        return dist, angle

    def direction_and_angle_to_quay(self):
        bearing, dist = utils.D2L(self.quay.colliding_edge, self.eta[0:2])
        angle = bearing - self.eta[-1]

        return dist, angle

    def close(self):
        print("===================================")
        print("-------- End of simulation --------")
        print("===================================")
        try:
            self.data["total time"] = sum(self.data["time"])
            self.data["num control intervals"] = len(self.data["time"])
            self.data["average time"] = np.mean(self.data["time"])
            self.data["time max"] = max(self.data["time"])
            self.data["time min"] = min(self.data["time"])
            self.data["time std deviation"] = np.std(self.data["time"])

            save_data(self.data, type(self).__name__)
            print("Data was collected")
            print("Report: ")
            print(f"Total time: {np.round(self.data['total time'], 5)}")
            print(f"Avg time:   {np.round(self.data['average time'], 5)}")
            print(
                f"Std time:   {np.round(self.data['time std deviation'], 5)}")
            print(f"Max time:   {np.round(self.data['time max'], 5)}")
            print(f"Min time:   {np.round(self.data['time min'], 5)}")
        except AttributeError:
            print("No data collected")

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
