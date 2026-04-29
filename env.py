"""
Proporciona una clase compatible con el gestor de contexto para que el resto del código 
pueda obtener/liberar entornos de Gymnasium sin dispersar bloques 
try/finally por todas partes.
"""

import gymnasium as gym
from individual import Individual


class CartPoleEnv:

    ENV_ID = "CartPole-v1"
    MAX_STEPS = 500        # límite estricto impuesto por el entorno

    def __init__(self, render_mode: str = None):
        self._env = gym.make(self.ENV_ID, render_mode=render_mode)

    # context manager 

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # Métodos públicos 

    def close(self):
        self._env.close()

    @property
    def unwrapped(self):
        return self._env

    def run_episode(self, individual: Individual, seed: int = None) -> float:
        """Ejecuta un episodio y devuelve la recompensa total (= pasos superados)."""
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
        """Devuelve la recompensa promedio en n_episodios episodios independientes."""
        rewards = [
            self.run_episode(individual, seed=None if seed is None else seed + i)
            for i in range(n_episodes)
        ]
        return sum(rewards) / len(rewards)
