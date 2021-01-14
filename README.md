# CoMaze Gym
An OpenAI Gym-compatible implementation of the zero-shot coordination and communication benchmark game CoMaze.

## Description

For an exhaustive description of the game, please refer to [this page](https://marieossenkopf.webnode.com/l/comaze-live-coding-session-at-emecom20/).


## Usage

`gym` must be installed. CoMaze environment can be created via running inside a `python` interpreter:

```python
>>> import gym
>>> import comaze_gym
>>> env = gym.make('CoMaze-7x7-Sparse-v0')
```

## Installation

### Installing via pip

This package is available in PyPi as `comaze-gym`

```bash
pip install comaze-gym
```

### Installing via cloning this repository

```bash
git clone https://www.github.com/Near32/comaze-gym
cd comaze-gym
pip install -e .
```
