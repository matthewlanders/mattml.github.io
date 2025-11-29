"""Modified from https://github.com/RDelg/rl-book/tree/master"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import List, Tuple
from scipy.ndimage import gaussian_filter1d


class Environment:
    def reset(self):
        raise NotImplementedError

    def step(self, action: int):
        raise NotImplementedError


class CliffGridWorld(Environment):
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.reset()

    def reset(self):
        self._ns = 0
        self._ms = 0
        return (self._ns, self._ms)

    def step(self, action: int):
        assert 0 <= action < 4, f"Invalid action {action}"
        self._ns += self.actions[action][0]
        self._ms += self.actions[action][1]
        self._ns = max(0, min(self._ns, self.n - 1))
        self._ms = max(0, min(self._ms, self.m - 1))
        reward = -1
        if 0 < self._ms < self.m - 1 and self._ns == 0:
            self._ns, self._ms = 0, 0
            reward = -100
        done = self._ns == 0 and self._ms == (self.m - 1)
        return done, (self._ns, self._ms), reward


class Agent:
    def __init__(self, n_states: Tuple[int, int], n_actions: int, gamma: float = 1.0):
        self.Q = np.zeros((n_states[0], n_states[1], n_actions))
        self.gamma = gamma

    def get_action(self, state: Tuple[int, int], epsilon: float) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(0, 4)
        return np.argmax(self.Q[state[0], state[1]])

    def get_max_q(self, state):
        return np.max(self.Q[state[0], state[1]])


class SARSAAgent(Agent):
    def learn(self, state, action, reward, next_state, next_action, alpha: float):
        current_q = self.Q[state[0], state[1], action]
        next_q = self.Q[next_state[0], next_state[1], next_action] if next_state is not None else 0
        self.Q[state[0], state[1], action] += alpha * (reward + self.gamma * next_q - current_q)


class QLearningAgent(Agent):
    def learn(self, state, action, reward, next_state, next_action, alpha: float):
        current_q = self.Q[state[0], state[1], action]
        next_q = self.get_max_q(next_state) if next_state is not None else 0
        self.Q[state[0], state[1], action] += alpha * (reward + self.gamma * next_q - current_q)


def run_episode(env, agent, epsilon: float, alpha: float):
    state = env.reset()
    total_reward = 0
    done = False
    action = agent.get_action(state, epsilon)

    while not done:
        done, next_state, reward = env.step(action)
        total_reward += reward
        next_action = agent.get_action(next_state, epsilon) if not done else None
        if not done:
            agent.learn(state, action, reward, next_state, next_action, alpha)
        state = next_state
        action = next_action if not done else None

    return total_reward


def train(agent_class, n_episodes: int = 500, n_runs: int = 100):
    env = CliffGridWorld(4, 12)
    rewards_history = []

    for _ in trange(n_runs, desc=f"Training {agent_class.__name__}"):
        agent = agent_class((4, 12), 4)
        episode_rewards = []

        for _ in range(n_episodes):
            total_reward = run_episode(env, agent, epsilon=0.1, alpha=0.5)
            episode_rewards.append(total_reward)

        rewards_history.append(episode_rewards)

    return np.mean(rewards_history, axis=0)


def plot_results(sarsa_rewards, qlearning_rewards):
    plt.figure(figsize=(8, 6))

    # Apply Gaussian smoothing
    sigma = 5  # Adjust this value to control smoothing amount
    sarsa_smooth = gaussian_filter1d(sarsa_rewards, sigma)
    qlearning_smooth = gaussian_filter1d(qlearning_rewards, sigma)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # Plot with new colors and no legend
    plt.plot(sarsa_smooth, color='black', linewidth=2)
    plt.plot(qlearning_smooth, color='darkgray', linewidth=2)

    plt.ylim(-100, 1)
    plt.ylabel('Sum of rewards during episodes', size=18)
    plt.xlabel('Episodes', size=18)
    plt.savefig('cliff_world_results.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    sarsa_rewards = train(SARSAAgent)
    qlearning_rewards = train(QLearningAgent)
    plot_results(sarsa_rewards, qlearning_rewards)
