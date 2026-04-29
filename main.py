import time
import numpy as np

from individual import Individual
from population import create_population, get_best, get_stats
from fitness import evaluate_population
from operators import (
    tournament_selection, roulette_selection,
    CROSSOVER_FNS, SELECTION_FNS,
    gaussian_mutation,
)


class GeneticAlgorithm:
    """Full generational GA for CartPole-v1.

    Parameters
    ----------
    population_size  : number of individuals per generation (>= 30)
    hidden_size      : neurons in the single hidden layer of the policy network
    n_generations    : how many generations to run (>= 50)
    mutation_rate    : probability that each gene is perturbed
    mutation_sigma   : standard deviation of Gaussian noise applied to mutated genes
    crossover_rate   : probability that two parents recombine (else copied)
    crossover_type   : "single_point" | "uniform" | "arithmetic"
    selection_type   : "tournament" | "roulette"
    tournament_k     : pool size for tournament selection
    elitism          : how many top individuals survive unchanged each generation
    n_episodes       : episodes averaged to compute each individual's fitness
    seed             : int for reproducibility; None for non-deterministic runs
    verbose          : print progress each generation
    """

    def __init__(
        self,
        population_size: int = 30,
        hidden_size: int = 4,
        n_generations: int = 50,
        mutation_rate: float = 0.10,
        mutation_sigma: float = 0.30,
        crossover_rate: float = 0.80,
        crossover_type: str = "single_point",
        selection_type: str = "tournament",
        tournament_k: int = 3,
        elitism: int = 2,
        n_episodes: int = 5,
        seed: int = None,
        verbose: bool = True,
    ):
        self.population_size = population_size
        self.hidden_size = hidden_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_rate = crossover_rate
        self.crossover_fn = CROSSOVER_FNS[crossover_type]
        self.selection_fn = SELECTION_FNS[selection_type]
        self.tournament_k = tournament_k
        self.elitism = elitism
        self.n_episodes = n_episodes
        self.seed = seed
        self.verbose = verbose

        self.history: dict = {"mean": [], "max": [], "min": [], "std": []}
        self.best_individual: Individual | None = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _select(self, population: list) -> Individual:
        if self.selection_fn is tournament_selection:
            return tournament_selection(population, k=self.tournament_k)
        return self.selection_fn(population)

    def _record(self, population: list) -> dict:
        stats = get_stats(population)
        for key in self.history:
            self.history[key].append(stats[key])
        return stats

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Evolve the population and return the fitness history."""
        if self.seed is not None:
            np.random.seed(self.seed)

        population = create_population(self.population_size, self.hidden_size)
        start = time.time()

        for gen in range(self.n_generations):
            # Evaluation
            eval_seed = None if self.seed is None else self.seed + gen * 1000
            evaluate_population(
                population,
                n_episodes=self.n_episodes,
                seed=eval_seed,
            )

            stats = self._record(population)

            # Track all-time best
            gen_best = get_best(population)
            if (self.best_individual is None
                    or gen_best.fitness > self.best_individual.fitness):
                self.best_individual = gen_best.clone()
                self.best_individual.fitness = gen_best.fitness

            if self.verbose:
                elapsed = time.time() - start
                print(
                    f"  Gen {gen + 1:3d}/{self.n_generations} | "
                    f"Mean: {stats['mean']:6.1f} | "
                    f"Max: {stats['max']:6.1f} | "
                    f"Std: {stats['std']:5.1f} | "
                    f"Time: {elapsed:.0f}s"
                )

            # Build next generation
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            next_gen: list = []

            # Elitism: carry forward the top individuals unchanged
            for i in range(min(self.elitism, self.population_size)):
                elite = population[i].clone()
                elite.fitness = population[i].fitness
                next_gen.append(elite)

            # Fill the rest with crossover + mutation
            while len(next_gen) < self.population_size:
                p1 = self._select(population)
                p2 = self._select(population)

                if np.random.random() < self.crossover_rate:
                    c1, c2 = self.crossover_fn(p1, p2)
                else:
                    c1, c2 = p1.clone(), p2.clone()

                c1 = gaussian_mutation(c1, self.mutation_rate, self.mutation_sigma)
                c2 = gaussian_mutation(c2, self.mutation_rate, self.mutation_sigma)

                next_gen.append(c1)
                if len(next_gen) < self.population_size:
                    next_gen.append(c2)

            population = next_gen

        return self.history


# ── convenience wrapper ───────────────────────────────────────────────────────

def run_experiment(config: dict, verbose: bool = True) -> dict:
    """Instantiate GeneticAlgorithm from a config dict and run it.

    Returns a dict with keys:
        history        – per-generation fitness stats
        best_fitness   – scalar fitness of the all-time best individual
        best_individual – the Individual object
        config         – the original config dict
    """
    # Strip display-only keys before passing to GA constructor
    ga_config = {k: v for k, v in config.items() if k not in ("name", "short_name")}
    ga = GeneticAlgorithm(**ga_config, verbose=verbose)

    if verbose and "name" in config:
        print(f"\n{'='*60}")
        print(f"  {config['name']}")
        print(f"{'='*60}")

    history = ga.run()
    return {
        "history":        history,
        "best_fitness":   ga.best_individual.fitness,
        "best_individual": ga.best_individual,
        "config":         config,
    }


# ── quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_experiment(
        {
            "name": "Smoke test",
            "population_size": 30,
            "n_generations": 5,
            "mutation_rate": 0.10,
            "crossover_type": "single_point",
            "seed": 42,
        },
        verbose=True,
    )
    print(f"\nBest fitness: {result['best_fitness']:.1f}")
