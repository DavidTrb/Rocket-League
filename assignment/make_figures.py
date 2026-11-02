"""Figures for the report: Optuna analysis + TensorBoard curves."""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from optuna_common import STORAGE

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ALGOS = ["sarsa", "expected_sarsa", "q_learning"]
NAMES = {"sarsa": "SARSA", "expected_sarsa": "Expected SARSA", "q_learning": "Q-Learning"}
COL = {"sarsa": "#1f77b4", "expected_sarsa": "#2ca02c", "q_learning": "#d62728"}
PARAMS = ["alpha", "gamma", "eps_start", "eps_end", "decay_frac", "q_init"]
studies = {a: optuna.load_study(study_name=a, storage=STORAGE) for a in ALGOS}
dfs = {a: s.trials_dataframe() for a, s in studies.items()}

# stats
for a in ALGOS:
  d = dfs[a]
  done = d[d.state == "COMPLETE"]
  print(a, "trials", len(d), "complete", len(done), "pruned", (d.state == "PRUNED").sum(),
        "best", done.value.max(), "median", done.value.median(), "worst", done.value.min())

# 1. optimization history
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for i, a in enumerate(ALGOS):
  d = dfs[a][dfs[a].state == "COMPLETE"]
  ax[i].scatter(d.number, d.value, s=14, color=COL[a], alpha=.6)
  ax[i].plot(d.number, d.value.cummax(), color="k", lw=1.5, label="best so far")
  ax[i].set_title(NAMES[a]); ax[i].set_xlabel("trial"); ax[i].set_ylim(-260, -40)
ax[0].set_ylabel("mean greedy return"); ax[0].legend()
plt.tight_layout(); plt.savefig("figures/history.png", dpi=150); plt.close()

# 2. importances
imp = {}
for a in ALGOS:
  imp[a] = optuna.importance.get_param_importances(studies[a])
  print(a, {k: round(v, 3) for k, v in imp[a].items()})
fig, ax = plt.subplots(figsize=(8, 3.6))
w = 0.27
for i, a in enumerate(ALGOS):
  ax.bar(np.arange(len(PARAMS)) + (i - 1) * w, [imp[a].get(p, 0) for p in PARAMS], w, color=COL[a], label=NAMES[a])
ax.set_xticks(range(len(PARAMS))); ax.set_xticklabels(PARAMS); ax.set_ylabel("importance (fANOVA)"); ax.legend()
plt.tight_layout(); plt.savefig("figures/importance.png", dpi=150); plt.close()

# 3. slice plots
fig, ax = plt.subplots(2, 3, figsize=(12, 6.6), sharey=True, constrained_layout=True)
for r, p in enumerate(["alpha", "q_init"]):
  for i, a in enumerate(ALGOS):
    d = dfs[a][dfs[a].state == "COMPLETE"]
    sc = ax[r, i].scatter(d["params_" + p], d.value, c=d.number, cmap="viridis", s=18)
    if p == "alpha": ax[r, i].set_xscale("log")
    ax[r, i].set_xlabel(p); ax[r, i].set_title(NAMES[a])
  ax[r, 0].set_ylabel("mean greedy return")
fig.colorbar(sc, ax=ax, label="trial number", shrink=.8)
plt.savefig("figures/slice.png", dpi=150, bbox_inches="tight"); plt.close()

# 4. contour-like: alpha vs eps_end for each algo coloured by value
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for i, a in enumerate(ALGOS):
  d = dfs[a][dfs[a].state == "COMPLETE"]
  sc = ax[i].scatter(d.params_alpha, d.params_gamma, c=d.value.clip(-200), cmap="RdYlGn", s=26, edgecolor="k", lw=.3)
  ax[i].set_xscale("log"); ax[i].set_xlabel("alpha"); ax[i].set_title(NAMES[a])
ax[0].set_ylabel("gamma"); fig.colorbar(sc, ax=ax, label="return (clipped at -200)", shrink=.9)
plt.savefig("figures/alpha_gamma.png", dpi=150, bbox_inches="tight"); plt.close()

# 5. tensorboard curves
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for i, a in enumerate(ALGOS):
  ea = EventAccumulator(f"runs/{a}"); ea.Reload()
  ev = ea.Scalars("eval/mean_return"); tr = ea.Scalars("train/mean_return_last_100")
  ax[i].plot([e.step for e in tr], [e.value for e in tr], color="gray", label="train (last 100, with exploration)")
  ax[i].plot([e.step for e in ev], [e.value for e in ev], color=COL[a], lw=2, label="eval (greedy, 100 ep.)")
  ax[i].set_title(NAMES[a]); ax[i].set_xlabel("episode"); ax[i].set_ylim(-250, 0)
  v = [e.value for e in ev]
  print(a, "eval first", v[0], "at 50k", v[4], "last", v[-1], "best", max(v), "std last 10", np.std(v[-10:]))
ax[0].set_ylabel("return"); ax[0].legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig("figures/tensorboard.png", dpi=150); plt.close()
for a in ALGOS:
  ea = EventAccumulator(f"runs/{a}"); ea.Reload()
  fig, ax = plt.subplots(figsize=(5, 3.2))
  ev = ea.Scalars("eval/mean_return"); tr = ea.Scalars("train/mean_return_last_100")
  ax.plot([e.step for e in tr], [e.value for e in tr], color="gray", label="train")
  ax.plot([e.step for e in ev], [e.value for e in ev], color=COL[a], lw=2, label="eval")
  ax.set_title(NAMES[a]); ax.set_xlabel("episode"); ax.set_ylabel("return"); ax.set_ylim(-250, 0); ax.legend(loc="lower right")
  plt.tight_layout(); plt.savefig(f"figures/tb_{a}.png", dpi=150); plt.close()
