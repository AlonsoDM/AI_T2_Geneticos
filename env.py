"""
env.py – Thin CartPole-v1 environment wrapper.

Provides a context-manager-friendly class so the rest of the code
can obtain/release Gymnasium environments without scattering
try/finally blocks everywhere.
"""

import gymnasium as gym
from individual import Individual


class CartPoleEnv:
    """Lightweight wrapper around a single CartPole-v1 Gymnasium environment."""

    ENV_ID = "CartPole-v1"
    MAX_STEPS = 500        # hard limit imposed by the environment

    def __init__(self, render_mode: str = None):
        self._env = gym.make(self.ENV_ID, render_mode=render_mode)

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── public API ────────────────────────────────────────────────────────────

    def close(self):
        self._env.close()

    @property
    def unwrapped(self):
        return self._env

    def run_episode(self, individual: Individual, seed: int = None) -> float:
        """Run one episode and return the total reward (= steps survived)."""
        obs, _ = self._env.reset(seed=seed)
        total_reward = 0.0
        done = False
        while not done:
            action = individual.act(obs)
            obs, reward, terminated, truncated, _ = self._env.step(action)
            total_reward += reward
            done = terminated or truncated
        return total_reward

    def evaluate(self, individual: Individual, n_episodes: int = 5,
                 seed: int = None) -> float:
        """Return average reward over n_episodes independent episodes."""
        rewards = [
            self.run_episode(individual, seed=None if seed is None else seed + i)
            for i in range(n_episodes)
        ]
        return sum(rewards) / len(rewards)
