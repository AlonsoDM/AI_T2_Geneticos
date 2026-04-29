import numpy as np
from individual import Individual


def create_population(size: int, hidden_size: int = 4) -> list:
    return [Individual(hidden_size=hidden_size) for _ in range(size)]


def get_best(population: list) -> Individual:
    """Devuelve al individuo con la mejor condición física (la condición física debe estar configurada)."""
    return max(population, key=lambda ind: ind.fitness if ind.fitness is not None else float("-inf"))


def get_stats(population: list) -> dict:
    """Devuelve la media/máximo/mínimo/desviación estándar de los valores de aptitud de la población."""
    fitnesses = [ind.fitness for ind in population if ind.fitness is not None]
    if not fitnesses:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
    arr = np.array(fitnesses, dtype=float)
    return {
        "mean": float(arr.mean()),
        "max":  float(arr.max()),
        "min":  float(arr.min()),
        "std":  float(arr.std()),
    }
