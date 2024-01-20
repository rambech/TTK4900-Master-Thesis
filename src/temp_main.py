"""
Main script for running Vehicle simulator

"""
from stable_baselines3 import PPO, TD3
import os
import numpy as np
from rl.env import ForwardDockingEnv, DPEnv, SidewaysDockingEnv
from maps import SimpleMap
from vehicle import Otter
import json

from utils import D2R

import pygame

env_type = "sideways"
render_mode = None
random_weather = False
seed = 0
timestep_multiplier = 5
threshold = 1
SECONDS = 120
VEHICLE_FPS = 60
RL_FPS = 20
# EPISODES = 10000
# TIMESTEPS = SECONDS*RL_FPS  # *timestep_multiplier
eta_init = np.array([-10, 0, 0, 0, 0, 0], float)
eta_d = np.array([25-0.75-1, 0, 0, 0, 0, 0], float)

# Initialize vehicle
vehicle = Otter(dt=1/VEHICLE_FPS)

map = SimpleMap()
if env_type == "docking":
    env = ForwardDockingEnv(vehicle, map, seed=seed, eta_init=eta_init,
                            render_mode=render_mode, FPS=RL_FPS)

if env_type == "sideways":
    env = SidewaysDockingEnv(vehicle, map, seed=seed,
                             render_mode=render_mode, FPS=RL_FPS)


"""
RL parameters
"""
model_type = "PPO"
folder_name = f"{model_type}-{env_type}-1-b"
episode = 10392000/2400
load_iteration = f"{int(episode*2400)}"  # "12000000"

models_dir = f"models"
model_path = f"{models_dir}/{folder_name}/{folder_name}_{load_iteration}_steps.zip"
assert (
    os.path.exists(model_path)
), f"{model_path} does not exist"

if model_type == "PPO":
    model = PPO.load(model_path, env=env)

elif model_type == "TD3":
    model = TD3.load(model_path, env=env)

# ---------------
# Data collection
# ---------------
test_dir = "test"
test_name = "sideways"
test_path = f"{models_dir}/{test_name}"

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

file_name = f"{test_name}.json"
test_path = os.path.join(test_dir, file_name)

data = {}

episodes = 100

num_episodes = 0
for ep in range(episodes):
    obs, _ = env.reset()
    terminated = False
    print(f"Obs: {obs}")
    num_steps = 0
    cum_reward = 0
    data.update({f"{num_episodes}": {}})
    north_pos = []
    east_pos = []
    psi = []
    while not terminated:
        action, _ = model.predict(obs)
        obs, reward, terminated, trunc, info = env.step(action)
        cum_reward += reward

        north_pos.append(info["eta"][0])
        east_pos.append(info["eta"][1])
        psi.append(info["eta"][2])

        if False:  # env_type == "docking":
            print(f"Observation: \n \
                    delta_x:    {obs[0]} \n \
                    delta_y:    {obs[1]} \n \
                    delta_psi:  {obs[2]} \n \
                    u:          {obs[3]} \n \
                    v:          {obs[4]} \n \
                    r:          {obs[5]} \n \
                    d_q         {obs[6]} \n \
                    psi_q       {obs[7]} \n \
                    d_c_w       {obs[8]} \n \
                    d_c_e       {obs[9]} \n")
        cum_reward += reward
        num_steps += 1

        print(f"Timestep: {num_steps}")
        print(f"Reward: {reward}")
        print(f"Cum reward: {cum_reward}")

    data[f"{num_episodes}"].update({"Total reward": cum_reward})
    data[f"{num_episodes}"].update(
        {"Termination state": info["Termination state"]})
    data[f"{num_episodes}"].update({"North pos": north_pos})
    data[f"{num_episodes}"].update({"East pos": east_pos})
    data[f"{num_episodes}"].update({"Psi": psi})

    num_episodes += 1

    with open(test_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)

if render_mode == "human":
    env.close()
