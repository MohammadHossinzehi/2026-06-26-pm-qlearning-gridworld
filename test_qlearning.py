import unittest

from qlearning import QLearningAgent


class TestQLearningAgent(unittest.TestCase):
    def test_update_formula(self):
        agent = QLearningAgent(actions=[0, 1], alpha=0.5, gamma=0.9, epsilon=0.0, seed=1)
        agent.q_table[("s2", 0)] = 4.0
        agent.q_table[("s2", 1)] = 2.0
        agent.update(state="s1", action=0, reward=1.0, next_state="s2", done=False)
        # target = reward + gamma * max(next_q) = 1.0 + 0.9 * 4.0 = 4.6
        # new_q = old_q + alpha * (target - old_q) = 0 + 0.5 * (4.6 - 0) = 2.3
        self.assertAlmostEqual(agent.q_table[("s1", 0)], 2.3)

    def test_update_terminal_ignores_future(self):
        agent = QLearningAgent(actions=[0, 1], alpha=1.0, gamma=0.9, epsilon=0.0, seed=1)
        agent.q_table[("s2", 0)] = 100.0
        agent.update(state="s1", action=1, reward=5.0, next_state="s2", done=True)
        self.assertAlmostEqual(agent.q_table[("s1", 1)], 5.0)

    def test_best_action_picks_max(self):
        agent = QLearningAgent(actions=[0, 1, 2], epsilon=0.0, seed=1)
        agent.q_table[("s", 0)] = 1.0
        agent.q_table[("s", 1)] = 5.0
        agent.q_table[("s", 2)] = 3.0
        self.assertEqual(agent.best_action("s"), 1)

    def test_choose_action_greedy_when_epsilon_zero(self):
        agent = QLearningAgent(actions=[0, 1], epsilon=0.0, seed=1)
        agent.q_table[("s", 0)] = 0.0
        agent.q_table[("s", 1)] = 10.0
        for _ in range(20):
            self.assertEqual(agent.choose_action("s"), 1)

    def test_choose_action_explores_when_epsilon_one(self):
        agent = QLearningAgent(actions=[0, 1], epsilon=1.0, seed=1)
        agent.q_table[("s", 0)] = 0.0
        agent.q_table[("s", 1)] = 10.0
        seen = {agent.choose_action("s") for _ in range(50)}
        self.assertEqual(seen, {0, 1})

    def test_train_returns_one_reward_per_episode(self):
        class DummyEnv:
            def reset(self):
                return "s"

            def step(self, action):
                return "s", 1.0, True

        agent = QLearningAgent(actions=[0, 1], epsilon=0.1, seed=1)
        rewards = agent.train(DummyEnv(), episodes=10, max_steps=5)
        self.assertEqual(len(rewards), 10)
        self.assertTrue(all(r == 1.0 for r in rewards))


if __name__ == "__main__":
    unittest.main()
