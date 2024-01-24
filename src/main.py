"""
Main script for running testing and running different RL environments

"""
import os
import numpy as np
from maps import SimpleMap
from vehicle import Otter

import argparse

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
# TODO: Add conventional control simulator

# ----------------
# Argument parsing
# ----------------
parser = argparse.ArgumentParser(
    description="Run Alexander's simplified simulator")

# General arguments
parser.add_argument("-e", "--env", type=str, default="forward", choices=["forward", "sideways"],
                    help="forward or sideways environment")
parser.add_argument("-m", "--manual", action="store_true",
                    help="choose manual control, works in rl and sim")

# RL related arguments
parser.add_argument("-rl", action="store_true",
                    help="run --env scenario as an rl environment")
parser.add_argument("-a", "--algo", type=str,
                    default="PPO", help="choose RL algorithm")
parser.add_argument("--model", type=str,
                    default="71-a", help="choose model name")
parser.add_argument("-i", "--iteration", type=int, default=18000000,
                    help="choose iteration number")

arguments = parser.parse_args()


# --------------------
# Simulation constants
# --------------------
env_type = arguments.env
random_weather = False
seed = None
timestep_multiplier = 5
threshold = 1
SECONDS = 120
VEHICLE_FPS = 60    # Vehicle dynamics frame rate
CONTROL_FPS = 20    # Control loop frame rate
eta_init = np.array([0, 0, 0, 0, 0, 0], float)
eta_d = np.array([25/2-0.75-1, 0, 0, 0, 0, 0], float)

# Initialize vehicle
vehicle = Otter(dt=1/VEHICLE_FPS)

if arguments.rl == True:
    map = SimpleMap()

    if env_type == "forward":
        from rl.env import ForwardDockingEnv
        env = ForwardDockingEnv(vehicle, map, seed=seed, eta_init=eta_init,
                                render_mode="human", FPS=CONTROL_FPS)

    if env_type == "sideways":
        from rl.env import SidewaysDockingEnv
        env = SidewaysDockingEnv(vehicle, map, seed=seed,
                                 render_mode="human", FPS=CONTROL_FPS)

    elif env_type == "DP":
        from rl.env import DPEnv
        env = DPEnv(vehicle, map, seed, eta_init=eta_init, render_mode="human",
                    FPS=CONTROL_FPS, threshold=threshold, random_weather=random_weather)

    if arguments.manual == False:
        """
        RL parameters
        """
        # Type of algorithm, PPO or TD3
        model_type = arguments.algo
        # Model to load
        folder_name = f"{model_type}-{env_type}-71-a"
        # Model iteration to load
        load_iteration = arguments.iteration

        models_dir = f"models"
        model_path = f"{models_dir}/{folder_name}/{folder_name}_{load_iteration}_steps.zip"
        assert (
            os.path.exists(model_path)
        ), f"{model_path} does not exist"

        if model_type == "PPO":
            from stable_baselines3 import PPO
            model = PPO.load(model_path, env=env)

        elif model_type == "TD3":
            from stable_baselines3 import TD3
            model = TD3.load(model_path, env=env)

        episodes = 10

        for ep in range(episodes):
            obs, _ = env.reset()
            terminated = False
            escape_sim = False
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
                        if event.key == K_ESCAPE:
                            terminated = True
                            escape_sim = True

                cunt += 1

            if escape_sim:
                break

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
            action = np.zeros(2, float)
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

else:
    from simulator import Simulator
    from control import Manual
    from maps import Target

    # Initialize simulator
    map = SimpleMap()
    target_position = Target(eta_d, vehicle, map.origin)
    control = Manual(dof=vehicle.dof)
    simulator = Simulator(vehicle, control, map, seed, target_position,
                          eta_init, CONTROL_FPS)
    simulator.simulate()
