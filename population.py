import numpy as np
from individual import Individual


def create_population(size: int, hidden_size: int = 4) -> list:
    """Return a list of `size` randomly initialised individuals."""
    return [Individual(hidden_size=hidden_size) for _ in range(size)]


def get_best(population: list) -> Individual:
    """Return the individual with the highest fitness (fitness must be set)."""
    return max(population, key=lambda ind: ind.fitness if ind.fitness is not None else float("-inf"))


def get_stats(population: list) -> dict:
    """Return mean/max/min/std of the population's fitness values."""
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
