"""
RL en


"""
# TODO: Add parametric costs theta_lambda and theta_v
# TODO: See what it takes to base this on torch
#       it might be better to actually follow Martinsen et. al for the estimator
#       References:
#           Combining system identification with reinforcement learning-based MPC
#           Reinforcement learning-based NMPC for tracking control of ASVs: Theory and experiments

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN


class ParamEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.terminated = False
        self.obs = None
        # TODO: Make action space
        # TODO: Make observation space

    def reset(self):
        ...

    def step(self):
        if self.obs is None:
            print("You must run update_obs(obs) before step()")
            raise TypeError

        observation = self.obs

        # Prediction Error
        error = observation[0] - observation[2]

        # Prediction Error method
        reward = -abs(error.T @ error)

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
    def __init__(self, parameters: dict, train=False, log_name="estimator") -> None:
        self.init_parameters = parameters
        self.parameters = parameters

        # =============================
        # Safe initial model parameters
        # =============================
        # TODO: Move this?
        # TODO: Determine safe parameter ranges
        if True:
            self.parameters = {
                # Safe ranges in the comments
                "m_total": 1,  # [50, 150]
                "xg": 1,    # [0.1, 1]
                "Iz": 1,    # ?
                "Xudot": 1,  # ?
                "Yvdot": 1,
                "Nrdot": 1,
                "Xu": 1,
                "Yv": 1,
                "Nr": 1,
                "k_port": 1,
                "k_std": 1,
                "w1": 1,
                "w2": 1,
                "w3": 1
            }

        self.env = ParamEnv()

        # Training parameters
        self.train = train
        if self.train:
            self.num_timesteps = 0
            self.log_interval = 4
            self.log_name = "logs/" + log_name

            self.model = DQN("MlpPolicy", env=self.env, verbose=1)

            # Establish callback for fetching best models
            from stable_baselines3.common.callbacks import CheckpointCallback

            self.checkpoint_callback = CheckpointCallback(
                save_freq=1000,    # Save every tenth episode
                save_path="models",
                name_prefix=log_name,
                save_replay_buffer=True,
                save_vecnormalize=True,
            )

        else:
            # TODO: Add models
            self.model = DQN.load("Insert model path here", self.env)

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
            self.model.learn(
                total_timesteps=1, callback=self.checkpoint_callback, reset_num_timesteps=False
            )
            parameters, _ = self.model.predict(obs)
        else:
            self.env.update_obs(obs)
            parameters, _ = self.model.predict(obs)

        self.parameters = {
            # Safe ranges in the comments
            "m_total": parameters[0],   # [50, 150]
            "xg":      parameters[1],   # [0.1, 1]
            "Iz":      parameters[2],   # ?
            "Xudot":   parameters[3],   # ?
            "Yvdot":   parameters[4],
            "Nrdot":   parameters[5],
            "Xu":      parameters[6],
            "Yv":      parameters[7],
            "Nr":      parameters[8],
            "k_port":  parameters[9],
            "k_std":   parameters[10],
            "w1":      parameters[11],
            "w2":      parameters[12],
            "w3":      parameters[13]
        }

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
