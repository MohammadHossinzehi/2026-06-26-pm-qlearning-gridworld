import unittest

from gridworld import GridWorld


class TestGridWorld(unittest.TestCase):
    def setUp(self):
        self.env = GridWorld(
            rows=3,
            cols=3,
            start=(0, 0),
            goal=(2, 2),
            obstacles=[(1, 1)],
            pits=[(0, 2)],
        )

    def test_reset_returns_start(self):
        self.assertEqual(self.env.reset(), (0, 0))

    def test_step_moves_in_bounds(self):
        self.env.reset()
        state, reward, done = self.env.step(1)  # down
        self.assertEqual(state, (1, 0))
        self.assertFalse(done)
        self.assertAlmostEqual(reward, -0.1)

    def test_step_blocked_by_wall(self):
        self.env.reset()
        self.env.step(1)  # to (1,0)
        state, _, _ = self.env.step(3)  # right into obstacle (1,1)
        self.assertEqual(state, (1, 0))  # stayed put

    def test_step_out_of_bounds_stays_put(self):
        self.env.reset()
        state, _, _ = self.env.step(0)  # up, already at top row
        self.assertEqual(state, (0, 0))

    def test_reaching_goal_is_terminal_with_bonus(self):
        self.env.state = (1, 2)
        state, reward, done = self.env.step(1)  # down into goal
        self.assertEqual(state, (2, 2))
        self.assertTrue(done)
        self.assertEqual(reward, 10.0)

    def test_falling_in_pit_is_terminal_with_penalty(self):
        self.env.state = (0, 1)
        state, reward, done = self.env.step(3)  # right into pit
        self.assertEqual(state, (0, 2))
        self.assertTrue(done)
        self.assertEqual(reward, -10.0)

    def test_step_after_terminal_raises(self):
        self.env.state = self.env.goal
        with self.assertRaises(RuntimeError):
            self.env.step(0)

    def test_all_states_excludes_obstacles(self):
        states = self.env.all_states()
        self.assertNotIn((1, 1), states)
        self.assertEqual(len(states), 3 * 3 - 1)


if __name__ == "__main__":
    unittest.main()
