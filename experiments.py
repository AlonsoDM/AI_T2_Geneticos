"""
experiments.py – Run 3 GA configurations on CartPole-v1 and produce plots.

Usage:
    python experiments.py

Output files (created in the working directory):
    results_config1.npy, results_config2.npy, results_config3.npy
    plot_individual_configs.png
    plot_comparison.png
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering (no display required)
import matplotlib.pyplot as plt

from main import run_experiment

# ── Experiment configurations ─────────────────────────────────────────────────

CONFIGS = [
    {
        # Small population, low mutation rate, single-point crossover.
        # Conservative exploration: fast convergence but risk of premature
        # convergence to local optima.
        "name":            "Config 1 – Conservative (Pop=30, mut=0.05, single-point)",
        "short_name":      "Config 1\n(Pop=30, mut=0.05,\nsingle-point)",
        "population_size": 30,
        "hidden_size":     4,
        "n_generations":   50,
        "mutation_rate":   0.05,
        "mutation_sigma":  0.30,
        "crossover_rate":  0.80,
        "crossover_type":  "single_point",
        "selection_type":  "tournament",
        "tournament_k":    3,
        "elitism":         2,
        "n_episodes":      5,
        "seed":            42,
    },
    {
        # Medium population, moderate mutation, uniform crossover.
        # Balanced exploration/exploitation; uniform crossover mixes genes
        # from both parents more thoroughly than single-point.
        "name":            "Config 2 – Balanced (Pop=50, mut=0.15, uniform)",
        "short_name":      "Config 2\n(Pop=50, mut=0.15,\nuniform)",
        "population_size": 50,
        "hidden_size":     4,
        "n_generations":   50,
        "mutation_rate":   0.15,
        "mutation_sigma":  0.30,
        "crossover_rate":  0.85,
        "crossover_type":  "uniform",
        "selection_type":  "tournament",
        "tournament_k":    5,
        "elitism":         2,
        "n_episodes":      5,
        "seed":            42,
    },
    {
        # Large population, high mutation, arithmetic (blend) crossover.
        # Aggressive diversity: slower to converge but less likely to get
        # stuck; arithmetic crossover naturally stays within the convex hull
        # of parent genes, reducing wild jumps.
        "name":            "Config 3 – Aggressive (Pop=60, mut=0.30, arithmetic)",
        "short_name":      "Config 3\n(Pop=60, mut=0.30,\narithmetic)",
        "population_size": 60,
        "hidden_size":     4,
        "n_generations":   50,
        "mutation_rate":   0.30,
        "mutation_sigma":  0.30,
        "crossover_rate":  0.90,
        "crossover_type":  "arithmetic",
        "selection_type":  "tournament",
        "tournament_k":    5,
        "elitism":         3,
        "n_episodes":      5,
        "seed":            42,
    },
]

COLORS = ["#2196F3", "#4CAF50", "#FF5722"]   # blue, green, deep-orange


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_single(ax, history: dict, title: str, color: str) -> None:
    """Draw mean ± std band and max fitness curve on ax."""
    gens = np.arange(1, len(history["mean"]) + 1)
    mean = np.array(history["mean"])
    maxi = np.array(history["max"])
    std  = np.array(history["std"])

    ax.fill_between(gens, mean - std, mean + std,
                    alpha=0.18, color=color, label="_std band")
    ax.plot(gens, mean, color=color, linewidth=2,   label="Mean fitness")
    ax.plot(gens, maxi, color=color, linewidth=2,
            linestyle="--", label="Max fitness")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (steps survived)")
    ax.set_ylim(0, 520)
    ax.axhline(500, color="gray", linewidth=0.8, linestyle=":", label="Max possible (500)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_individual_configs(results: list, out_path: str = "plot_individual_configs.png") -> None:
    """One subplot per configuration showing mean/max fitness over generations."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Genetic Algorithm – CartPole-v1\nFitness per Generation by Configuration",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, res, color in zip(axes, results, COLORS):
        _plot_single(ax, res["history"], res["config"]["name"], color)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_comparison(results: list, out_path: str = "plot_comparison.png") -> None:
    """Overlay all three configurations' max and mean fitness for direct comparison."""
    gens = np.arange(1, len(results[0]["history"]["mean"]) + 1)

    fig, (ax_max, ax_mean) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Configuration Comparison – CartPole-v1",
                 fontsize=13, fontweight="bold")

    for res, color in zip(results, COLORS):
        label = res["config"]["short_name"].replace("\n", " ")
        ax_max.plot(gens, res["history"]["max"],
                    color=color, linewidth=2, label=label)
        ax_mean.plot(gens, res["history"]["mean"],
                     color=color, linewidth=2, label=label)

    for ax, title in zip((ax_max, ax_mean), ("Max Fitness", "Mean Fitness")):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (steps survived)")
        ax.set_ylim(0, 520)
        ax.axhline(500, color="gray", linewidth=0.8, linestyle=":", label="Theoretical max")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_bar_summary(results: list, out_path: str = "plot_bar_summary.png") -> None:
    """Bar chart comparing the best fitness achieved by each configuration."""
    labels = [f"Config {i+1}" for i in range(len(results))]
    best   = [r["best_fitness"] for r in results]
    final_mean = [r["history"]["mean"][-1] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, best,       width, label="Best fitness (all-time)", color=COLORS)
    bars2 = ax.bar(x + width / 2, final_mean, width, label="Mean fitness (last gen)",
                   color=[c + "88" for c in COLORS])   # semi-transparent version

    ax.set_title("Best vs. Final Mean Fitness per Configuration",
                 fontweight="bold")
    ax.set_ylabel("Fitness (steps survived)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 540)
    ax.axhline(500, color="gray", linewidth=0.8, linestyle=":", label="Max possible")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main entry point ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CartPole-v1 Genetic Algorithm Experiments")
    print("=" * 60)
    total_start = time.time()

    results = []
    for cfg in CONFIGS:
        t0 = time.time()
        res = run_experiment(cfg, verbose=True)
        elapsed = time.time() - t0
        print(f"\n  → Best fitness: {res['best_fitness']:.1f}  "
              f"(finished in {elapsed:.0f}s)\n")
        results.append(res)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for i, res in enumerate(results):
        h = res["history"]
        print(f"  Config {i+1}: best={res['best_fitness']:.1f}  "
              f"final_mean={h['mean'][-1]:.1f}  "
              f"final_max={h['max'][-1]:.1f}")

    print(f"\n  Total time: {(time.time() - total_start):.0f}s")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_individual_configs(results)
    plot_comparison(results)
    plot_bar_summary(results)
    print("Done.")


if __name__ == "__main__":
    main()
