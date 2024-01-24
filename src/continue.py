import os
import time
import json
import numpy as np
from rl.env import ForwardDockingEnv, DPEnv, SidewaysDockingEnv
from vehicle import Otter
from maps import SimpleMap
from stable_baselines3.common.callbacks import CheckpointCallback

# Training settings
model_type = "PPO"
env_type = "docking"
random_weather = False
seed = 1
threshold = 3
timestep_multiplier = 5
SECONDS = 120
VEHICLE_FPS = 60
RL_FPS = 20
EPISODES = 7500
TIMESTEPS = SECONDS*RL_FPS*EPISODES  # *timestep_multiplier

# test_name = input(f"Test name is {model_type}-{env_type}-")
# test_name = f"{model_type}-{env_type}-{test_name}"

models_dir = "models"
test_name = "PPO-docking-71-c"
load_iteration = "11856000"
log_dir = "logs"
model_path = f"{models_dir}/{test_name}"
log_path = f"{log_dir}/{test_name}"
model_path = f"{models_dir}/{test_name}/{test_name}_{load_iteration}_steps.zip"

# Initialize vehicle
vehicle = Otter(dt=1/VEHICLE_FPS)

map = SimpleMap()

# User input
if env_type == "docking":
    eta_init = np.array([0, 0, 0, 0, 0, 0], float)
    env = ForwardDockingEnv(vehicle, map, seed=seed,
                            render_mode=None, FPS=RL_FPS)

if env_type == "sideways":
    eta_init = np.array([0, 0, 0, 0, 0, 0], float)
    env = SidewaysDockingEnv(vehicle, map, seed=seed,
                             render_mode=None, FPS=RL_FPS)

elif env_type == "DP":
    eta_init = np.array([-10, 0, 0, 0, 0, 0], float)
    env = DPEnv(vehicle, map, seed, render_mode=None,
                FPS=RL_FPS, threshold=threshold, random_weather=random_weather)

checkpoint_callback = CheckpointCallback(
    save_freq=RL_FPS*SECONDS*10,    # Save every tenth episode
    save_path=model_path,
    name_prefix=test_name,
    save_replay_buffer=True,
    save_vecnormalize=True,
)

env.reset(seed)
if model_type == "PPO":
    from stable_baselines3 import PPO

    model = PPO.load(model_path, env=env)

if model_type == "TD3":
    from stable_baselines3 import TD3

    model = TD3.load(model_path, env=env)


model.learn(total_timesteps=TIMESTEPS,
            callback=checkpoint_callback, tb_log_name=model_type)
