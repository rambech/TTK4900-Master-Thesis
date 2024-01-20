from stable_baselines3.common.env_checker import check_env
from rl.env import ForwardDockingEnv
from maps import SimpleMap, Target
from vehicle import Otter
import numpy as np

# Initialize constants
fps = 50
eta_init = np.array([0, 0, 0, 0, 0, 0], float)
# eta_d = np.array([15-0.75-1, 0, 0, 0, 0, 0], float)

# Initialize vehicle
vehicle = Otter(dt=1/fps)

map = SimpleMap()
# target = Target(eta_d, vehicle.L, vehicle.B, vehicle.scale, map.origin)
env = ForwardDockingEnv(vehicle, map)

# It will check your custom environment and output additional warnings if needed
check_env(env)
