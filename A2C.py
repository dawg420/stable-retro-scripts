"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import argparse

import os
import gymnasium as gym
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import A2C
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

import retro
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor





class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


# def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
#     if state is None:
#         state = retro.State.DEFAULT
#     env = retro.make(game, state, **kwargs)
#     env = StochasticFrameSkip(env, n=4, stickprob=0.25)
#     if max_episode_steps is not None:
#         env = TimeLimit(env, max_episode_steps=max_episode_steps)
#     return env

def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state,render_mode="rgb_array", **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env



def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Airstriker-Genesis")
    # Use like this: --game="WWFArcade-Genesis"
    
    parser.add_argument("--state", default=retro.State.DEFAULT)
    # Use like this: --state='VeryEasy_Yokozuna-01'
    #States you can use:
    #     game_states = [
    #     'VeryEasy_Yokozuna-01',
    #     'VeryEasy_Yokozuna-02',
    #     'VeryEasy_Yokozuna-03',
    #     'VeryEasy_Yokozuna-04',
    #     'VeryEasy_Yokozuna-05',
    #     'VeryEasy_Yokozuna-06',
    #     'VeryEasy_Yokozuna-07'
    # ]

    # game_states_veryhard = [
    #     'VeryHard_Yokozuna-01',
    #     'VeryHard_Yokozuna-02',
    #     'VeryHard_Yokozuna-03',
    #     'VeryHard_Yokozuna-04',
    #     'VeryHard_Yokozuna-05',
    #     'VeryHard_Yokozuna-06',
    #     'VeryHard_Yokozuna-07'
    # ]
    
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    def make_env_easy():
        env = make_retro(game=args.game, state="VeryEasy_Yokozuna-03", scenario=args.scenario)
        env = wrap_deepmind_retro(env)
        env = Monitor(env)  # Add this line to wrap the environment with Monitor
        return env
    def make_env_hard():
        env = make_retro(game=args.game, state="VeryHard_Yokozuna-03", scenario=args.scenario)
        env = wrap_deepmind_retro(env)
        env = Monitor(env)  # Add this line to wrap the environment with Monitor
        return env

    train_env = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env_easy] * 8), n_stack=4)) 
    eval_env = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env_easy] * 8), n_stack=4)) 
 
    # Setup TensorBoard logging 
    easy_log_dir = os.path.join(".", "logs", "easy_train")
 
    # Callbacks for evaluation and saving the best model 
    eval_callback = EvalCallback( 
        eval_env, 
        best_model_save_path=easy_log_dir, 
        log_path=easy_log_dir, 
        eval_freq=500,  # frequency of evaluations 
        deterministic=True, 
        render=False 
    ) 
 
    model = A2C( 
        policy="CnnPolicy", 
        env=train_env, 
        learning_rate=lambda f: f * 2.5e-4, 
        n_steps=128, 
        batch_size=32, 
        n_epochs=4, 
        gamma=0.99, 
        gae_lambda=0.95, 
        clip_range=0.1, 
        ent_coef=0.01, 
        verbose=1, 
        tensorboard_log=easy_log_dir 
    ) 
 
    model.learn( 
        total_timesteps=200000,
        log_interval=1, 
        callback=eval_callback  # Attach evaluation callback 
    )
    
    train_env.close() 
    eval_env.close()
    
    train_env = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env_hard] * 8), n_stack=4)) 
    eval_env = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env_hard] * 8), n_stack=4)) 
    
    # Setup TensorBoard logging 
    hard_log_dir = os.path.join(".", "logs", "hard_train")
 
    # Callbacks for evaluation and saving the best model 
    eval_callback = EvalCallback( 
        eval_env, 
        best_model_save_path=hard_log_dir, 
        log_path=hard_log_dir, 
        eval_freq=500,  # frequency of evaluations 
        deterministic=True, 
        render=False 
    )
    
    prev_model_path = os.path.join(easy_log_dir, "best_model")
    model = A2C.load(os.path.expanduser(prev_model_path), env=train_env, tensorboard_log=hard_log_dir)
    model.learn( 
        total_timesteps=200000, 
        log_interval=1, 
        callback=eval_callback  # Attach evaluation callback 
    )
 
    train_env.close() 
    eval_env.close()


if __name__ == "__main__":
    main()
