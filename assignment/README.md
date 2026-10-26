# Tabular methods on Racetrack (Assignment 1, Option A)

SARSA, Expected SARSA and Q-learning on a Gymnasium version of the Racetrack
environment (Sutton & Barto, ch. 5), with hyperparameter search in Optuna.

The parent folder (`environment.py`, `racetracks.py`, `constants.py`) comes from
Ondrej Biza (MIT license, see LICENSE_racetrack_MIT).
Everything else is ours.

## Setup
    pip install -r requirements.txt

## Run (in this order, from this folder: racetrack_rl/assignment)
    python optuna_sarsa.py            # study "sarsa"
    python optuna_expected_sarsa.py   # study "expected_sarsa"
    python optuna_q_learning.py       # study "q_learning"
    python long_training.py           # needs the 3 studies above
    tensorboard --logdir runs

Run the 3 Optuna scripts one after another, or start them a few seconds apart
if in parallel (they share optuna_racetrack.db and SQLite fails if all three
create the tables at the same instant).

## Files
- racetrack_gym.py: Gymnasium wrapper (Discrete obs and actions)
- tabular_agents.py: the three algorithms, epsilon schedule, evaluation
- optuna_common.py: search space and objective shared by the 3 scripts
