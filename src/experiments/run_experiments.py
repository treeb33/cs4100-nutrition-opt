"""
Run comparison experiments and save plots to results/.

Usage (from repo root, with venv active):
    python -m src.experiments.run_experiments --csv data/RAW_recipes.csv

Plots saved to results/:
    convergence.png           — best score per iteration for each algorithm
    best_scores.png           — mean best score ± std across seeds
    constraint_satisfaction.png — feasibility rate per algorithm
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.recipe import load_recipes, curate_pool
from src.meal_plan import MealPlan
from src.evaluation.objective import score_plan, DEFAULT_PREFS, is_feasible
from src.search.hill_climbing import hill_climb_with_restarts
from src.search.simulated_annealing import simulated_annealing_multi
from src.search.genetic import run_genetic


# ---------------------------------------------------------------------------
# Convergence trackers — thin wrappers that record score history
# ---------------------------------------------------------------------------

def _hc_with_history(pool, user_prefs, rng, n_restarts=5, max_steps=500):
    """Hill climbing: record best score after every step across all restarts."""
    best_score = float("-inf")
    history = []

    for _ in range(n_restarts):
        current = MealPlan.random(pool_size=len(pool), rng=rng)
        current_score = score_plan(current, pool, user_prefs)
        for _ in range(max_steps):
            neighbor = current.neighbor(pool_size=len(pool), rng=rng)
            neighbor_score = score_plan(neighbor, pool, user_prefs)
            if neighbor_score > current_score:
                current = neighbor
                current_score = neighbor_score
            best_score = max(best_score, current_score)
            history.append(best_score)

    return history


def _sa_with_history(pool, user_prefs, rng, n_runs=5, max_steps=500,
                     initial_temp=1.0, cooling_rate=0.995, min_temp=1e-4):
    """SA: record best-so-far after every step across all runs."""
    import math
    best_score = float("-inf")
    history = []

    for _ in range(n_runs):
        current = MealPlan.random(pool_size=len(pool), rng=rng)
        current_score = score_plan(current, pool, user_prefs)
        temp = initial_temp
        for _ in range(max_steps):
            if temp < min_temp:
                break
            neighbor = current.neighbor(pool_size=len(pool), rng=rng)
            neighbor_score = score_plan(neighbor, pool, user_prefs)
            delta = neighbor_score - current_score
            if delta > 0 or rng.random() < math.exp(delta / temp):
                current = neighbor
                current_score = neighbor_score
            best_score = max(best_score, current_score)
            history.append(best_score)
            temp *= cooling_rate

    return history


def _ga_with_history(pool, user_prefs, rng, pop_size=30, n_generations=50):
    """GA: record best score per generation."""
    from src.meal_plan import NUM_SLOTS
    from src.search.genetic import _pick_parent, _combine, _apply_mutation

    pool_size = len(pool)
    pop = [MealPlan.random(pool_size=pool_size, rng=rng) for _ in range(pop_size)]
    best_score = float("-inf")
    history = []

    for _ in range(n_generations):
        fitness = [score_plan(p, pool, user_prefs) for p in pop]
        gen_best = max(fitness)
        best_score = max(best_score, gen_best)
        history.append(best_score)

        top_idx = max(range(pop_size), key=lambda i: fitness[i])
        next_gen = [pop[top_idx]]
        while len(next_gen) < pop_size:
            mom = _pick_parent(pop, fitness, rng=rng)
            dad = _pick_parent(pop, fitness, rng=rng)
            child_a, child_b = _combine(mom, dad, rng=rng)
            child_a = _apply_mutation(child_a, pool_size, rng=rng)
            child_b = _apply_mutation(child_b, pool_size, rng=rng)
            next_gen.extend([child_a, child_b])
        pop = next_gen[:pop_size]

    return history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "Hill Climbing":       "#4C72B0",
    "Simulated Annealing": "#DD8452",
    "Genetic Algorithm":   "#55A868",
}


def plot_convergence(histories: dict[str, list[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, hist in histories.items():
        xs = np.linspace(0, 100, len(hist))
        ax.plot(xs, hist, label=label, color=COLORS[label], linewidth=2)
    ax.set_xlabel("% of total iterations", fontsize=12)
    ax.set_ylabel("Best score (higher is better)", fontsize=12)
    ax.set_title("Convergence curves — all algorithms", fontsize=13)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_best_scores(
    mean_scores: dict[str, float],
    std_scores: dict[str, float],
    out_path: Path,
) -> None:
    labels = list(mean_scores.keys())
    means  = [mean_scores[l] for l in labels]
    stds   = [std_scores[l]  for l in labels]
    colors = [COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, means, yerr=stds, color=colors, edgecolor="white",
                  width=0.5, capsize=6)
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Mean best score ± std", fontsize=12)
    ax.set_title("Best score per algorithm (across seeds)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_constraint_satisfaction(csr: dict[str, float], out_path: Path) -> None:
    labels = list(csr.keys())
    vals   = [csr[l] * 100 for l in labels]
    colors = [COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f}%",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylim(0, 115)
    ax.set_ylabel("Feasibility rate (%)", fontsize=12)
    ax.set_title("Constraint satisfaction rate per algorithm", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        default="data/RAW_recipes.csv")
    parser.add_argument("--pool-size",  type=int,   default=400)
    parser.add_argument("--seeds",      type=int,   nargs="+", default=[42, 7, 100, 2024, 31])
    parser.add_argument("--hc-steps",   type=int,   default=500)
    parser.add_argument("--sa-steps",   type=int,   default=500)
    parser.add_argument("--ga-gens",    type=int,   default=50)
    parser.add_argument("--n-restarts", type=int,   default=5,
                        help="restarts for HC / runs for SA")
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # ---- load + curate pool ----
    print(f"\nLoading recipes from {args.csv} ...")
    raw = load_recipes(args.csv)
    print(f"  {len(raw)} recipes after time filter")
    pool = curate_pool(raw, target_size=args.pool_size, rng=random.Random(args.seeds[0]))
    print(f"  curated pool: {len(pool)} recipes")

    user_prefs = dict(DEFAULT_PREFS)

    # ---- multi-seed runs for score + feasibility stats ----
    print(f"\nRunning {len(args.seeds)} seeds × 3 algorithms ...")
    algo_configs = [
        ("Hill Climbing",       hill_climb_with_restarts, {"n_restarts": args.n_restarts, "max_steps": args.hc_steps}),
        ("Simulated Annealing", simulated_annealing_multi, {"n_runs": args.n_restarts, "max_steps": args.sa_steps}),
        ("Genetic Algorithm",   run_genetic,              {"pop_size": 30, "n_generations": args.ga_gens}),
    ]

    mean_scores = {}
    std_scores  = {}
    csr         = {}

    for label, fn, kwargs in algo_configs:
        scores    = []
        feasibles = []
        print(f"\n  {label}")
        for seed in args.seeds:
            rng = random.Random(seed)
            t0 = time.perf_counter()
            plan, score = fn(pool, user_prefs, rng=rng, **kwargs)
            elapsed = time.perf_counter() - t0
            feasible = is_feasible(plan, pool, user_prefs)
            scores.append(score)
            feasibles.append(feasible)
            print(f"    seed={seed}  score={score:.4f}  feasible={feasible}  ({elapsed:.1f}s)")
        mean_scores[label] = float(np.mean(scores))
        std_scores[label]  = float(np.std(scores))
        csr[label]         = sum(feasibles) / len(feasibles)

    # ---- single convergence run (seed[0]) ----
    print(f"\nRecording convergence curves (seed={args.seeds[0]}) ...")
    histories = {}
    convergence_fns = [
        ("Hill Climbing",       _hc_with_history, {"n_restarts": args.n_restarts, "max_steps": args.hc_steps}),
        ("Simulated Annealing", _sa_with_history, {"n_runs": args.n_restarts, "max_steps": args.sa_steps}),
        ("Genetic Algorithm",   _ga_with_history, {"pop_size": 30, "n_generations": args.ga_gens}),
    ]
    for label, fn, kwargs in convergence_fns:
        rng = random.Random(args.seeds[0])
        histories[label] = fn(pool, user_prefs, rng=rng, **kwargs)
        print(f"  {label}: {len(histories[label])} points")

    # ---- save plots ----
    print("\nSaving plots to results/ ...")
    plot_convergence(histories,           results_dir / "convergence.png")
    plot_best_scores(mean_scores, std_scores, results_dir / "best_scores.png")
    plot_constraint_satisfaction(csr,     results_dir / "constraint_satisfaction.png")

    # ---- print summary table ----
    print("\n" + "─" * 62)
    print(f"{'Algorithm':<22} {'Mean score':<16} {'Feasibility'}")
    print("─" * 62)
    for label in mean_scores:
        print(f"{label:<22} {mean_scores[label]:>+.4f} ± {std_scores[label]:.4f}   {csr[label]*100:.0f}%")
    print("─" * 62)
    print("\nDone.")


if __name__ == "__main__":
    main()