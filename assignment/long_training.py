"""Longer training run with the best hyperparameters of each algorithm (logged to TensorBoard)."""
import json

import optuna
from tensorboardX import SummaryWriter

from optuna_common import STORAGE, TRACK
from racetrack_gym import RacetrackEnv
from tabular_agents import ALGOS, evaluate, train

LONG_EPISODES = 500_000
EVAL_EVERY = 10_000
SEED = 123

if __name__ == "__main__":
  summary = {}
  for algo in ALGOS:
    study = optuna.load_study(study_name=algo, storage=STORAGE)
    params = study.best_params
    print(algo, study.best_value, params, flush=True)
    env = RacetrackEnv(TRACK)
    writer = SummaryWriter(f"runs/{algo}")
    Q, _ = train(env, algo, n_episodes=LONG_EPISODES, seed=SEED, eval_every=EVAL_EVERY,
                 eval_episodes=100, writer=writer, **params)
    final = evaluate(env, Q, 1000)
    writer.add_scalar("eval/final_1000_episodes", final, LONG_EPISODES)
    writer.close()
    summary[algo] = {"params": params, "optuna_best": study.best_value, "final_eval_1000": final}
    print(algo, "final eval", final, flush=True)
  json.dump(summary, open("long_training_summary.json", "w"), indent=2)
