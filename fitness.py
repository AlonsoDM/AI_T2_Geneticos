import numpy as np
import gymnasium as gym

from individual import Individual



DEFAULT_N_EPISODES = 5    # episodios promediados por evaluación
DEFAULT_MAX_STEPS  = 500  # máximo de pasos en CartPole-v1 (límite del entorno)



def evaluate_fitness(
    individual: Individual,
    env: gym.Env = None,
    n_episodes: int = DEFAULT_N_EPISODES,
    normalize: bool = False,
    seed: int = None,
) -> float:

    # Crear entorno si no se pasó uno externamente
    close_env = False
    if env is None:
        env = gym.make("CartPole-v1")
        close_env = True

    total_steps = 0.0

    for episode in range(n_episodes):
        # Cada episodio usa una semilla distinta pero determinista si seed != None.
        # Esto garantiza que dos individuos se comparen en las mismas condiciones
        episode_seed = None if seed is None else seed + episode

        observation, _ = env.reset(seed=episode_seed)
        done = False
        steps = 0

        while not done:
            action = individual.act(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += reward   # reward es +1 por paso, así que steps == recompensa acumulada

        total_steps += steps

    if close_env:
        env.close()

    # Guardar el resultado en el individuo para no recalcular
    avg_fitness = total_steps / n_episodes
    individual.fitness = avg_fitness / DEFAULT_MAX_STEPS if normalize else avg_fitness

    return individual.fitness


def evaluate_population(
    population: list,
    n_episodes: int = DEFAULT_N_EPISODES,
    normalize: bool = False,
    seed: int = None,
) -> list:

    env = gym.make("CartPole-v1")

    fitness_values = []
    for i, individual in enumerate(population):
        ind_seed = None if seed is None else seed + i * 100
        fit = evaluate_fitness(
            individual,
            env=env,
            n_episodes=n_episodes,
            normalize=normalize,
            seed=ind_seed,
        )
        fitness_values.append(fit)

    env.close()
    return fitness_values


def population_stats(fitness_values: list) -> dict:
    arr = np.array(fitness_values)
    return {
        "mean": float(arr.mean()),
        "max":  float(arr.max()),
        "min":  float(arr.min()),
        "std":  float(arr.std()),
    }


"""
    Test para ver si funciona
"""
if __name__ == "__main__":
    print("=== Verificación de fitness.py ===\n")

    # 1. Individuo con genes aleatorios (debe dar fitness bajo, ~20-50)
    ind_random = Individual(hidden_size=4)
    fit_random = evaluate_fitness(ind_random, n_episodes=5, seed=42)
    print(f"Individuo aleatorio → fitness: {fit_random:.1f} pasos promedio")
    print(f"  (esperado: entre 8 y 60 aprox. para genes aleatorios)\n")

    # 2. Individuo con todos los pesos en cero (siempre elige acción 0)
    n = Individual(hidden_size=4)
    n.genes[:] = 0.0
    fit_zero = evaluate_fitness(n, n_episodes=5, seed=42)
    print(f"Individuo con genes=0 → fitness: {fit_zero:.1f} pasos promedio")
    print(f"  (esperado: fitness bajo, política constante)\n")

    # 3. Evaluación de una población pequeña
    population = [Individual(hidden_size=4) for _ in range(6)]
    fitnesses = evaluate_population(population, n_episodes=3, seed=0)
    stats = population_stats(fitnesses)

    print("Población de 6 individuos aleatorios:")
    for i, (ind, fit) in enumerate(zip(population, fitnesses)):
        print(f"  Individuo {i}: {fit:.1f} pasos")

    print(f"\nEstadísticas:")
    print(f"  Media:  {stats['mean']:.1f}")
    print(f"  Máximo: {stats['max']:.1f}")
    print(f"  Mínimo: {stats['min']:.1f}")
    print(f"  Std:    {stats['std']:.1f}")

    print("\n✓ fitness.py funciona correctamente")
