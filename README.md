# Q-Learning Grid World

A from-scratch implementation of tabular Q-learning, trained on a grid world with walls and a pit. No machine learning libraries involved — just plain Python, a dictionary for the Q-table, and the core Bellman update rule.

## What it does

An agent starts in the top-left corner of a 5x5 grid and has to learn how to reach the goal in the bottom-right corner, while avoiding a pit that gives a large penalty and walls it can't pass through. It doesn't know the map in advance. Through repeated trial and error (epsilon-greedy exploration), it updates a table of action-values (Q-values) using the standard Q-learning update:

```
Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
```

After enough episodes, the greedy policy derived from the Q-table reliably finds a short, safe path to the goal.

This is useful as a minimal, readable reference for how reinforcement learning actually works under the hood — most tutorials either hide the math behind a library or skip the engineering (environment design, terminal-state handling, exploration/exploitation). This project keeps both visible in under 150 lines of core logic.

## Project layout

- `gridworld.py` — the `GridWorld` environment: state transitions, rewards, rendering of the grid and learned policy as ASCII art.
- `qlearning.py` — the `QLearningAgent`: Q-table, epsilon-greedy action selection, the update rule, and a training loop.
- `main.py` — wires the two together: builds a 5x5 grid, trains for 500 episodes, prints the learned policy, and runs a greedy rollout to confirm the agent reaches the goal.
- `test_gridworld.py`, `test_qlearning.py` — unit tests for both modules.

## How to run it

Requires only Python 3 (no dependencies, no `pip install`).

```bash
python3 main.py
```

Example output:

```
Grid (S=start, G=goal, X=pit, #=wall):
S . . . .
. # . X .
. # . . .
. . . # .
. . . . G

Trained for 500 episodes.
Average reward, first 20 episodes:  -1.30
Average reward, last 20 episodes:   9.05

Learned policy:
v < < > v
v # v X v
v # v > v
> v v # v
> > > > G

Greedy rollout reached goal: True
Path length: 8 steps, total reward: 9.30
```

To run the test suite:

```bash
python3 -m unittest discover -v
```

All 14 tests should pass.

## Design decisions

- **Pure Python, no NumPy.** The state/action space here is tiny (25 states x 4 actions), so a `defaultdict` is plenty fast and keeps the dependency list at zero. It also makes the Bellman update completely transparent — you can read `qlearning.py` top to bottom without needing to know any array-library API.
- **Terminal states with mixed rewards.** The goal gives `+10`, the pit gives `-10`, and every other step costs `-0.1`. The step cost is what pushes the agent toward *shorter* paths rather than just *any* path to the goal — without it, wandering forever (while avoiding the pit) would never be penalized.
- **Walls as no-ops, not failures.** Walking into a wall or the grid boundary just leaves the agent in place (with the usual step penalty), rather than ending the episode. This is closer to how a real robot/agent would experience "I can't go that way" and avoids artificially terminating exploration.
- **Seeded randomness.** Both the environment's training loop and the agent's exploration accept a `seed`, so `main.py`'s output is fully reproducible.
- **Testing strategy.** `test_gridworld.py` checks the environment in isolation (movement, walls, terminal rewards) without involving the agent. `test_qlearning.py` checks the agent's update math against hand-computed expected values (see the comments in `test_update_formula`) and verifies exploration/exploitation behavior at the epsilon extremes (0.0 and 1.0), plus a training smoke test against a trivial dummy environment. Together they cover the environment and the learning algorithm independently, so a bug in one doesn't mask a bug in the other.
