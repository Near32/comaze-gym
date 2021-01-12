from .env import *
from .utils import *

import gym
from gym.envs.registration import register

env_dict = gym.envs.registration.registry.env_specs.copy()

for env in env_dict:
    if 'CoMaze' in env:
        del gym.envs.registration.registry.env_specs[env]

register(
    id='CoMaze-7x7-Sparse-v0',
    entry_point='comaze_gym.env:CoMazeGymEnv7x7Sparse'
)

register(
    id='CoMaze-7x7-Dense-v0',
    entry_point='comaze_gym.env:CoMazeGymEnv7x7Dense'
)