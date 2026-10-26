"""Gymnasium wrapper around the Racetrack environment (Sutton & Barto, ch. 5).

Environment dynamics: Racetrack by Ondrej Biza (MIT license)
Observation: Discrete(rows * cols * 5 * 5) -> index of (row, col, v_row, v_col)
Action:      Discrete(9) -> acceleration in {-1, 0, 1}^2
"""
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# environment.py, racetracks.py and constants.py live in the parent folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import environment  # noqa: E402
import racetracks  # noqa: E402

NUM_SPEEDS = 5
ACCELERATIONS = [(1, 1), (0, 1), (1, 0), (0, 0), (-1, 0), (0, -1), (1, -1), (-1, 1), (-1, -1)]


class RacetrackEnv(gym.Env):
  metadata = {"render_modes": []}

  def __init__(self, track="track_2", strict=False, displacement_prob=0.5, max_steps=300):
    self.grid = racetracks.TRACKS[track]
    self.rows, self.cols = self.grid.shape
    self.strict = strict
    self.max_steps = max_steps
    if strict:
      self.core = environment.RacetrackStrict(self.grid)
    else:
      self.core = environment.Racetrack(self.grid, random_displacement_probability=displacement_prob)

    self.observation_space = spaces.Discrete(self.rows * self.cols * NUM_SPEEDS * NUM_SPEEDS)
    self.action_space = spaces.Discrete(len(ACCELERATIONS))
    self.steps = 0

  def _obs(self):
    r, c, vr, vc = self.core.get_state()
    r = min(max(r, 0), self.rows - 1)
    c = min(max(c, 0), self.cols - 1)
    return ((r * self.cols + c) * NUM_SPEEDS + vr) * NUM_SPEEDS + vc

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    if seed is not None:
      import random
      random.seed(seed)
      np.random.seed(seed)
    self.core.reset()
    self.steps = 0
    return self._obs(), {}

  def step(self, action):
    reward = self.core.act(*ACCELERATIONS[int(action)])
    self.steps += 1
    terminated = bool(self.core.done)
    truncated = (not terminated) and self.steps >= self.max_steps
    return self._obs(), float(reward), terminated, truncated, {}
