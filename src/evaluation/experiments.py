"""comparison harness for the three local search algorithms.

drives hill climbing, simulated annealing, genetic search on the same
pool / prefs / seeds and reports solution quality, convergence speed, wall-clock, and
constraint satisfaction rate.
"""
from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass

from src.evaluation.objective import is_feasible
from src.recipe import Recipe
from src.search.genetic import run_genetic
from src.search.hill_climbing import hill_climb_with_restarts
from src.search.simulated_annealing import simulated_annealing_multi


@dataclass
class RunResult:
    seed: int
    score: float
    elapsed_s: float
    feasible: bool


@dataclass
class AlgoSummary:
    name: str
    runs: list[RunResult]

    @property
    def mean_score(self) -> float:
        return statistics.mean(r.score for r in self.runs)

    @property
    def std_score(self) -> float:
        return statistics.pstdev(r.score for r in self.runs) if len(self.runs) > 1 else 0.0

    @property
    def mean_time_s(self) -> float:
        return statistics.mean(r.elapsed_s for r in self.runs)

    @property
    def feasibility_rate(self) -> float:
        return sum(1 for r in self.runs if r.feasible) / len(self.runs)


def _run_one(
    algo_fn,
    pool: list[Recipe],
    user_prefs: dict,
    seed: int,
    **kwargs,
) -> RunResult:
    """run a single algorithm with a seeded RNG and time it.

    all three algorithms share the (pool, user_prefs, rng=...) signature
    so the harness can drive them through one call shape.
    """
    rng = random.Random(seed)
    t0 = time.perf_counter()
    plan, score = algo_fn(pool, user_prefs, rng=rng, **kwargs)
    elapsed = time.perf_counter() - t0
    return RunResult(
        seed=seed,
        score=score,
        elapsed_s=elapsed,
        feasible=is_feasible(plan, pool, user_prefs),
    )


def compare_algorithms(
    pool: list[Recipe],
    user_prefs: dict,
    seeds: list[int],
    *,
    hc_kwargs: dict | None = None,
    sa_kwargs: dict | None = None,
    ga_kwargs: dict | None = None,
) -> dict[str, AlgoSummary]:
    """
    run all three algorithms across the same seed list, return a summary
    per algorithm. Per-algorithm kwargs let the caller match comparable
    compute budgets (e.g. roughly equal total neighbor evaluations).
    """
    hc_kwargs = hc_kwargs or {"n_restarts": 5, "max_steps": 1000}
    sa_kwargs = sa_kwargs or {"n_runs": 5, "max_steps": 1000}
    ga_kwargs = ga_kwargs or {"pop_size": 30, "n_generations": 50}

    algorithms = [
        ("hill_climbing", hill_climb_with_restarts, hc_kwargs),
        ("simulated_annealing", simulated_annealing_multi, sa_kwargs),
        ("genetic", run_genetic, ga_kwargs),
    ]

    summaries: dict[str, AlgoSummary] = {}
    for name, algo_fn, kwargs in algorithms:
        print(f"\n=== {name} ===")
        runs = []
        for seed in seeds:
            print(f"  seed={seed} ...", end="", flush=True)
            result = _run_one(algo_fn, pool, user_prefs, seed, **kwargs)
            runs.append(result)
            print(
                f" score={result.score:.4f} "
                f"time={result.elapsed_s:.2f}s "
                f"feasible={result.feasible}"
            )
        summaries[name] = AlgoSummary(name=name, runs=runs)

    return summaries


def print_comparison_table(summaries: dict[str, AlgoSummary]) -> None:
    """Render a compact comparison table to stdout."""
    header = f"{'algorithm':<22} {'score (mean ± std)':<28} {'time (s)':<12} {'feasibility':<12}"
    print("\n" + header)
    print("-" * len(header))
    for name, s in summaries.items():
        score_col = f"{s.mean_score:>+.3f} ± {s.std_score:.3f}"
        print(
            f"{name:<22} {score_col:<28} {s.mean_time_s:<12.2f} "
            f"{s.feasibility_rate * 100:>5.1f}%"
        )


if __name__ == "__main__":
    # Small comparison run:
    #   python -m src.evaluation.experiments
    # Tune n_seeds / step counts via the kwargs below for a longer run.
    from src.recipe import load_recipes

    pool = load_recipes("data/RAW_recipes.csv")
    print(f"Loaded {len(pool)} recipes")

    user_prefs = {"calorie_target": 2000, "allergens": []}
    seeds = [42, 7, 100, 2024, 31]

    summaries = compare_algorithms(
        pool,
        user_prefs,
        seeds,
        hc_kwargs={"n_restarts": 3, "max_steps": 500},
        sa_kwargs={"n_runs": 3, "max_steps": 500},
        ga_kwargs={"pop_size": 20, "n_generations": 25},
    )
    print_comparison_table(summaries)
