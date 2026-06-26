import random
from collections import defaultdict


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=0.1, seed=None):
        self.actions = list(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(float)
        self.rng = random.Random(seed)

    def _q(self, state, action):
        return self.q_table[(state, action)]

    def best_action(self, state):
        best_value = max(self._q(state, a) for a in self.actions)
        best_actions = [a for a in self.actions if self._q(state, a) == best_value]
        return self.rng.choice(best_actions)

    def choose_action(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.best_action(state)

    def update(self, state, action, reward, next_state, done):
        current = self._q(state, action)
        future = 0.0 if done else max(self._q(next_state, a) for a in self.actions)
        target = reward + self.gamma * future
        self.q_table[(state, action)] = current + self.alpha * (target - current)

    def train(self, env, episodes=500, max_steps=100):
        rewards_per_episode = []
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0.0
            for _ in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            rewards_per_episode.append(total_reward)
        return rewards_per_episode

    def policy(self, env):
        return {state: self.best_action(state) for state in env.all_states()}
