"""Shared Optuna objective for the Racetrack tabular-methods experiment."""
import sys

import optuna

from racetrack_gym import RacetrackEnv
from tabular_agents import evaluate, train

TRACK = "track_2"
N_EPISODES = 40_000
REPORT_EVERY = 10_000
N_TRIALS = 100
STORAGE = "sqlite:///optuna_racetrack.db"

SEARCH_SPACE = {
  "alpha": "float log [0.01, 1.0]",
  "gamma": "float [0.90, 1.0]",
  "eps_start": "float [0.1, 1.0]",
  "eps_end": "float log [0.001, 0.2]",
  "decay_frac": "float [0.1, 0.9]",
  "q_init": "float step 10 [-100, 20]",
}


def suggest(trial):
  return dict(
    alpha=trial.suggest_float("alpha", 0.01, 1.0, log=True),
    gamma=trial.suggest_float("gamma", 0.90, 1.0),
    eps_start=trial.suggest_float("eps_start", 0.1, 1.0),
    eps_end=trial.suggest_float("eps_end", 0.001, 0.2, log=True),
    decay_frac=trial.suggest_float("decay_frac", 0.1, 0.9),
    q_init=trial.suggest_float("q_init", -100, 20, step=10),
  )


def run_study(algo, n_trials=N_TRIALS):
  env = RacetrackEnv(TRACK)

  def objective(trial):
    params = suggest(trial)

    def cb(ep, ev):
      trial.report(ev, ep)
      if trial.should_prune():
        raise optuna.TrialPruned()

    Q, _ = train(env, algo, n_episodes=N_EPISODES, seed=0, eval_every=REPORT_EVERY,
                 eval_episodes=50, callback=cb, **params)
    return evaluate(env, Q, 200)

  study = optuna.create_study(
    study_name=algo, storage=STORAGE, load_if_exists=True, direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=1))
  done = len(study.trials)
  study.optimize(objective, n_trials=max(0, n_trials - done))
  print(algo, "best", study.best_value, study.best_params)
