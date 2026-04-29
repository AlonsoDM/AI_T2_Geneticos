import numpy as np
from individual import Individual


# Selección

def tournament_selection(population: list, k: int = 3) -> Individual:
    """Return the fittest individual from k randomly sampled candidates."""
    candidates = np.random.choice(len(population), size=k, replace=False)
    return max((population[i] for i in candidates), key=lambda ind: ind.fitness)


def roulette_selection(population: list) -> Individual:
    """Fitness-proportional (roulette-wheel) selection."""
    fitnesses = np.array([ind.fitness for ind in population], dtype=float)
    fitnesses -= fitnesses.min() - 1e-6   # shift so all values are positive
    probs = fitnesses / fitnesses.sum()
    idx = np.random.choice(len(population), p=probs)
    return population[idx]


# Crossover 

def single_point_crossover(parent1: Individual, parent2: Individual) -> tuple:
    """Cut both chromosomes at one random point and swap tails."""
    n = len(parent1.genes)
    point = np.random.randint(1, n)
    g1 = np.concatenate([parent1.genes[:point], parent2.genes[point:]])
    g2 = np.concatenate([parent2.genes[:point], parent1.genes[point:]])
    return (
        Individual(hidden_size=parent1.hidden_size, genes=g1),
        Individual(hidden_size=parent1.hidden_size, genes=g2),
    )


def uniform_crossover(parent1: Individual, parent2: Individual, p: float = 0.5) -> tuple:
    """Each gene is independently drawn from either parent with probability p."""
    mask = np.random.random(len(parent1.genes)) < p
    g1 = np.where(mask, parent1.genes, parent2.genes)
    g2 = np.where(mask, parent2.genes, parent1.genes)
    return (
        Individual(hidden_size=parent1.hidden_size, genes=g1),
        Individual(hidden_size=parent1.hidden_size, genes=g2),
    )


def arithmetic_crossover(parent1: Individual, parent2: Individual) -> tuple:
    """Blend crossover: children are complementary linear combinations of parents."""
    alpha = np.random.random()
    g1 = alpha * parent1.genes + (1.0 - alpha) * parent2.genes
    g2 = (1.0 - alpha) * parent1.genes + alpha * parent2.genes
    return (
        Individual(hidden_size=parent1.hidden_size, genes=g1),
        Individual(hidden_size=parent1.hidden_size, genes=g2),
    )


# Mutación

def gaussian_mutation(
    individual: Individual,
    mutation_rate: float,
    sigma: float = 0.3,
) -> Individual:
    """Add Gaussian noise N(0, sigma) to each gene with probability mutation_rate."""
    genes = individual.genes.copy()
    mask = np.random.random(len(genes)) < mutation_rate
    genes[mask] += np.random.randn(int(mask.sum())) * sigma
    return Individual(hidden_size=individual.hidden_size, genes=genes)


# Registro (utilizado por GeneticAlgorithm) 

CROSSOVER_FNS = {
    "single_point": single_point_crossover,
    "uniform":      uniform_crossover,
    "arithmetic":   arithmetic_crossover,
}

SELECTION_FNS = {
    "tournament": tournament_selection,
    "roulette":   roulette_selection,
}
