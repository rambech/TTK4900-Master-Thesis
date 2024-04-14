"""
RL en


"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN


class ParamEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.terminated = False

    def reset(self):
        ...

    def step(self):
        observation = self.obs
        reward = -abs(observation[0] - observation[2])

        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def update_obs(self, obs):
        self.obs = obs

    def set_terminated(self):
        if self.terminated == False:
            self.terminated = True
        else:
            self.terminated = False


class Estimator():
    def __init__(self, parameters: dict, train=False) -> None:
        self.init_parameters = parameters
        self.parameters = parameters

        # Training parameters
        self.train = train
        if self.train:
            self.num_timesteps = 0
            self.log_interval = 4

        self.env = ParamEnv()
        self.model = DQN("MlpPolicy", env=self.env, verbose=1)

    def reset(self):
        self.parameters = self.init_parameters

    def step(self, pred_x, pred_u, act_x, act_u):
        """
        Step function for RL estimator

        Parameters
        ----------
            pred_x : np.array
                List of N predicted states opt_x_1
            pred_u : np.array
                List of N given control signals opt_u_1
            act_x : np.array
                List of N actually achieved states x_1
            act_u : np.array
                List of N actual thruster forces u_1

        Returns
        -------
            parameters : dict
                Dictionary of parameters for updating 
                nmpc model.

        """

        obs = np.concatenate([pred_x, pred_u, act_x, act_u])

        if self.train:
            # self.parameters = self.learn(obs)
            self.env.update_obs(obs)
            self.model.learn(total_timesteps=1, reset_num_timesteps=False)
            self.model.predict(obs)
        else:
            self.parameters = self.model.predict(obs)

        return self.parameters

    def stop(self):
        self.env.set_terminated()

    def learn(self, obs):
        """
        Step function for RL estimator

        Parameters
        ----------
            obs: list
                pred_x : np.array
                    List of N predicted states opt_x_1
                pred_u : np.array
                    List of N given control signals opt_u_1
                act_x : np.array
                    List of N actually achieved states x_1
                act_u : np.array
                    List of N actual thruster forces u_1

        Returns
        -------
            parameters : dict
                Dictionary of parameters for updating 
                nmpc model.

        """
        callback = None

        # First update from simulator
        self.env.update_obs(obs)

        # Then train

        total_timesteps, callback = self.model._setup_learn(
            total_timesteps,
            self.callback,
            self.reset_num_timesteps,
            self.tb_log_name,
            self.progress_bar,
        )

        rollout = self.model.collect_rollouts(
            self.env,
            train_freq=self.model.train_freq,
            action_noise=self.model.action_noise,
            callback=callback,
            learning_starts=self.model.learning_starts,
            replay_buffer=self.model.replay_buffer,
            log_interval=self.log_interval,
        )

        # if rollout.continue_training is False:
        #     break

        if self.num_timesteps > 0 and self.num_timesteps > self.model.learning_starts:
            # If no `gradient_steps` is specified,
            # do as many gradients steps as steps performed during the rollout
            gradient_steps = self.model.gradient_steps if self.model.gradient_steps >= 0 else rollout.episode_timesteps
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                self.train(batch_size=self.model.batch_size,
                           gradient_steps=gradient_steps)

        # callback.on_training_end()

        parameters, _ = self.model.predict(obs)

        return parameters

    def _pred_step(self, pred_x, pred_u, act_x, act_u):
        ...
