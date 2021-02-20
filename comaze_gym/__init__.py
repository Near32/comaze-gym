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

register(
    id='CoMaze-7x7-Dense-FixedActions-v0',
    entry_point='comaze_gym.env:CoMazeGymEnv7x7DenseFixedActions'
)

register(
    id='CoMaze-7x7-Dense-SinglePlayer-v0',
    entry_point='comaze_gym.env:CoMazeGymEnv7x7DenseSinglePlayer'
)

register(
    id='CoMaze-11x11-Sparse-v0',
    entry_point='comaze_gym.env:CoMazeGymEnv11x11Sparse'
)

register(
    id='CoMaze-11x11-Dense-v0',
    entry_point='comaze_gym.env:CoMazeGymEnv11x11Dense'
)