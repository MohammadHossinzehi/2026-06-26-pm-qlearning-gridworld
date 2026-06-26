ACTIONS = (0, 1, 2, 3)  # up, down, left, right
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}
ACTION_ARROWS = {0: "^", 1: "v", 2: "<", 3: ">"}


class GridWorld:
    """A rectangular grid environment for tabular reinforcement learning.

    The agent starts at `start`, receives `step_reward` for every move,
    and the episode ends when it reaches `goal` (+goal_reward) or a cell
    in `pits` (+pit_reward). `obstacles` are impassable walls.
    """

    def __init__(self, rows, cols, start, goal, obstacles=None, pits=None,
                 step_reward=-0.1, goal_reward=10.0, pit_reward=-10.0):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles or [])
        self.pits = set(pits or [])
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.pit_reward = pit_reward
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state

    def in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_terminal(self, pos):
        return pos == self.goal or pos in self.pits

    def all_states(self):
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.obstacles
        ]

    def step(self, action):
        if self.is_terminal(self.state):
            raise RuntimeError("step() called on a terminal state; call reset() first")
        dr, dc = ACTION_DELTAS[action]
        candidate = (self.state[0] + dr, self.state[1] + dc)
        if self.in_bounds(candidate) and candidate not in self.obstacles:
            self.state = candidate

        if self.state == self.goal:
            reward = self.goal_reward
        elif self.state in self.pits:
            reward = self.pit_reward
        else:
            reward = self.step_reward

        done = self.is_terminal(self.state)
        return self.state, reward, done

    def render(self, agent_pos=None):
        agent_pos = agent_pos or self.state
        lines = []
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                pos = (r, c)
                if pos == agent_pos:
                    row_chars.append("A")
                elif pos in self.obstacles:
                    row_chars.append("#")
                elif pos == self.goal:
                    row_chars.append("G")
                elif pos in self.pits:
                    row_chars.append("X")
                elif pos == self.start:
                    row_chars.append("S")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        return "\n".join(lines)

    def render_policy(self, policy):
        lines = []
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                pos = (r, c)
                if pos in self.obstacles:
                    row_chars.append("#")
                elif pos == self.goal:
                    row_chars.append("G")
                elif pos in self.pits:
                    row_chars.append("X")
                else:
                    row_chars.append(ACTION_ARROWS[policy[pos]])
            lines.append(" ".join(row_chars))
        return "\n".join(lines)
