"""
Main script for running testing and running different RL environments

"""
from stable_baselines3 import PPO, TD3
import os
import numpy as np
from rl.env import ForwardDockingEnv, DPEnv, SidewaysDockingEnv
from maps import SimpleMap
from vehicle import Otter

from utils import D2R

import pygame

# Keystroke inputs
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    K_TAB,
    KEYDOWN,
    K_q,
    K_w,
    K_a,
    K_s,
    QUIT,
)

# TODO: Make it possible to add disturbances using keystrokes,
#       with side arrows determining direction and up and down
#       determining the magnitude
# TODO: Make default models for running docking. I.e. when running
#       forward use 71-a 18000000, when running sideways use the other one
# TODO: Add command line parsing to run the script with greater ease
# TODO: Add conventional control simulator

# To test RL or not to test RL that is the question
RL = True

env_type = "forward"
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
eta_d = np.array([25-0.75-0.504, 0, 0, 0, 0, 0], float)

# Initialize vehicle
vehicle = Otter(dt=1/VEHICLE_FPS)

map = SimpleMap()
if env_type == "forward":
    env = ForwardDockingEnv(vehicle, map, seed=seed, eta_init=eta_init,
                            render_mode="human", FPS=RL_FPS)

if env_type == "sideways":
    env = SidewaysDockingEnv(vehicle, map, seed=seed,
                             render_mode="human", FPS=RL_FPS)

elif env_type == "DP":
    env = DPEnv(vehicle, map, seed, eta_init=eta_init, render_mode="human",
                FPS=RL_FPS, threshold=threshold, random_weather=random_weather)

if RL == True:
    """
    RL parameters
    """
    model_type = "PPO"
    folder_name = f"{model_type}-{env_type}-71-a"
    episode = 18000000/2400
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

    episodes = 10

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = False
        print(f"Obs: {obs}")
        cunt = 0
        cum_reward = 0
        while not terminated:
            action, _ = model.predict(obs)
            obs, reward, terminated, trunc, info = env.step(action)
            cum_reward += reward

            if env_type == "DP":
                print(f"Observation: \n \
                        delta_x:    {obs[0]} \n \
                        delta_y:    {obs[1]} \n \
                        delta_psi:  {obs[2]} \n \
                        u:          {obs[3]} \n \
                        v:          {obs[4]} \n \
                        r:          {obs[5]} \n")

            if env_type == "docking":
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
            print(f"Timestep: {cunt}")
            print(f"Reward: {reward}")
            print(f"Cum reward: {cum_reward}")

            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_TAB:
                        terminated = True

            cunt += 1

    env.close()

else:
    """
    Standard simulation parameters
    """

    episodes = 10

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = False
        print(f"Obs: {obs}")
        cunt = 0
        cum_reward = 0
        action = np.zeros(2, float)  # [-1, 1]
        while not terminated:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    # Quit if escape key is pressed
                    if event.key == K_ESCAPE:
                        terminated = True

                    # Manual forwards
                    if event.key == K_UP:
                        # Constant positive surge force
                        action = np.ones(2, float)

                    elif event.key == K_DOWN:
                        # Constant negative surge force
                        action = -np.ones(2, float)

                    elif event.key == K_RIGHT:
                        # Constant positive yaw moment
                        action = np.array([1, -1])

                    elif event.key == K_LEFT:
                        # Constant positive yaw moment
                        action = np.array([-1, 1])

                    elif event.key == K_q:
                        action = np.array([1, 0])

                    elif event.key == K_w:
                        action = np.array([0, 1])

                    elif event.key == K_a:
                        action = np.array([-1, 0])

                    elif event.key == K_s:
                        action = np.array([0, -1])

                else:
                    action = np.zeros(2, float)  # [-1, 1]

            obs, reward, terminated, trunc, info = env.step(action)
            if env_type == "DP":
                print(f"Observation: \n \
                        delta_x:    {obs[0]} \n \
                        delta_y:    {obs[1]} \n \
                        delta_psi:  {obs[2]} \n \
                        u:          {obs[3]} \n \
                        v:          {obs[4]} \n \
                        r:          {obs[5]} \n")

            if env_type == "docking":
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
            print(f"Timestep: {cunt}")
            print(f"Reward: {reward}")
            print(f"Cum reward: {cum_reward}")
            cunt += 1

    env.close()
