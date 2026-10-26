"""Tabular TD control: SARSA, Expected SARSA and Q-learning with epsilon-greedy exploration."""
import numpy as np

ALGOS = ("sarsa", "expected_sarsa", "q_learning")


def epsilon_at(ep, n_episodes, eps_start, eps_end, decay_frac):
  """Linear decay from eps_start to eps_end during the first decay_frac of training."""
  progress = min(1.0, ep / max(1.0, decay_frac * n_episodes))
  return eps_start + progress * (eps_end - eps_start)


def eps_greedy(q_row, eps, rng):
  if rng.random() < eps:
    return int(rng.integers(len(q_row)))
  best = np.flatnonzero(q_row == q_row.max())
  return int(best[0]) if len(best) == 1 else int(rng.choice(best))


def evaluate(env, Q, episodes=100):
  returns = []
  for _ in range(episodes):
    s, _ = env.reset()
    done, total = False, 0.0
    while not done:
      s, r, term, trunc, _ = env.step(int(np.argmax(Q[s])))
      total += r
      done = term or trunc
    returns.append(total)
  return float(np.mean(returns))


def train(env, algo, alpha, gamma, eps_start, eps_end, decay_frac, q_init, n_episodes,
          seed=0, eval_every=None, eval_episodes=100, callback=None, writer=None):
  """Train a tabular agent. callback(ep, eval_return) is called every eval_every episodes."""
  assert algo in ALGOS
  rng = np.random.default_rng(seed)
  Q = np.full((env.observation_space.n, env.action_space.n), q_init, dtype=np.float64)
  n_actions = env.action_space.n
  env.reset(seed=seed)
  train_returns = []

  for ep in range(n_episodes):
    eps = epsilon_at(ep, n_episodes, eps_start, eps_end, decay_frac)
    s, _ = env.reset()
    a = eps_greedy(Q[s], eps, rng)
    done, total = False, 0.0
    while not done:
      s2, r, term, trunc, _ = env.step(a)
      total += r
      done = term or trunc
      a2 = eps_greedy(Q[s2], eps, rng) if not term else 0
      if term:
        target = r
      elif algo == "sarsa":
        target = r + gamma * Q[s2, a2]
      elif algo == "q_learning":
        target = r + gamma * Q[s2].max()
      else:  # expected sarsa under the epsilon-greedy policy
        q2 = Q[s2]
        greedy = np.flatnonzero(q2 == q2.max())
        probs = np.full(n_actions, eps / n_actions)
        probs[greedy] += (1 - eps) / len(greedy)
        target = r + gamma * float(probs @ q2)
      Q[s, a] += alpha * (target - Q[s, a])
      s, a = s2, a2
    train_returns.append(total)

    if eval_every and (ep + 1) % eval_every == 0:
      ev = evaluate(env, Q, eval_episodes)
      if writer is not None:
        writer.add_scalar("eval/mean_return", ev, ep + 1)
        writer.add_scalar("train/epsilon", eps, ep + 1)
        writer.add_scalar("train/mean_return_last_100", float(np.mean(train_returns[-100:])), ep + 1)
      if callback is not None:
        callback(ep + 1, ev)
  return Q, train_returns
