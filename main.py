"""Train a tabular Q-learning agent to solve a grid world and report results."""
from gridworld import GridWorld, ACTIONS
from qlearning import QLearningAgent


def build_default_env():
    return GridWorld(
        rows=5,
        cols=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles=[(1, 1), (2, 1), (3, 3)],
        pits=[(1, 3)],
    )


def moving_average(values, window=20):
    if len(values) < window:
        return values
    return [
        sum(values[i - window:i]) / window
        for i in range(window, len(values) + 1)
    ]


def rollout(env, agent, max_steps=50):
    state = env.reset()
    path = [state]
    total_reward = 0.0
    for _ in range(max_steps):
        action = agent.best_action(state)
        state, reward, done = env.step(action)
        total_reward += reward
        path.append(state)
        if done:
            break
    return path, total_reward


def main():
    env = build_default_env()
    agent = QLearningAgent(ACTIONS, alpha=0.2, gamma=0.95, epsilon=0.2, seed=42)

    print("Grid (S=start, G=goal, X=pit, #=wall):")
    print(env.render())
    print()

    episodes = 500
    rewards = agent.train(env, episodes=episodes, max_steps=100)

    print(f"Trained for {episodes} episodes.")
    print(f"Average reward, first 20 episodes:  {sum(rewards[:20]) / 20:.2f}")
    print(f"Average reward, last 20 episodes:   {sum(rewards[-20:]) / 20:.2f}")
    print()

    policy = agent.policy(env)
    print("Learned policy:")
    print(env.render_policy(policy))
    print()

    path, total_reward = rollout(env, agent)
    reached_goal = path[-1] == env.goal
    print(f"Greedy rollout reached goal: {reached_goal}")
    print(f"Path length: {len(path) - 1} steps, total reward: {total_reward:.2f}")
    print(f"Path: {path}")


if __name__ == "__main__":
    main()
