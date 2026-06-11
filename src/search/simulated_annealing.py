"""Simulated annealing for weekly meal plan optimization."""
from __future__ import annotations

import math
import random

from src.meal_plan import MealPlan
from src.evaluation.objective import score_plan


def simulated_annealing(
    pool,
    user_prefs,
    rng: random.Random | None = None,
    max_steps: int = 10_000,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.995,
    min_temp: float = 1e-4,
):
    """Single simulated annealing run from a random start.

    At each step, sample one neighbor (single meal swap). Always accept
    improvements; accept worse moves with probability exp(delta / T),
    where delta is negative. T cools geometrically each step.

    The "best ever seen" plan is tracked separately from the current plan,
    so accepting a worse move late in the schedule doesn't lose progress.

    Returns (best_plan, best_score).
    """
    if rng is None:
        rng = random.Random()

    current = MealPlan.random(pool_size=len(pool), rng=rng)
    current_score = score_plan(current, pool, user_prefs)

    best_plan = current
    best_score = current_score

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
            if current_score > best_score:
                best_score = current_score
                best_plan = current

        temp *= cooling_rate

    return best_plan, best_score


def simulated_annealing_multi(
    pool,
    user_prefs,
    rng: random.Random | None = None,
    n_runs: int = 10,
    max_steps: int = 10_000,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.995,
    min_temp: float = 1e-4,
):
    """Run simulated_annealing n_runs times, return the best overall.

    Mirrors hill_climb_with_restarts so the experiment harness can drive
    both algorithms with the same shape of call.

    Returns (best_plan, best_score).
    """
    if rng is None:
        rng = random.Random()

    best_plan = None
    best_score = float("-inf")

    for i in range(n_runs):
        plan, score = simulated_annealing(
            pool,
            user_prefs,
            rng=rng,
            max_steps=max_steps,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            min_temp=min_temp,
        )
        print(f"  Run {i + 1}/{n_runs}: score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_plan = plan

    return best_plan, best_score


if __name__ == "__main__":
    # Smoke test against real loaded recipes:
    #   python -m src.search.simulated_annealing data/RAW_recipes.csv
    import sys

    from src.recipe import load_recipes

    pool = load_recipes(sys.argv[1])
    print(f"Loaded {len(pool)} recipes")

    user_prefs = {
        "calorie_target": 2000,
        "allergens": [],
    }

    rng = random.Random(42)
    plan, score = simulated_annealing_multi(
        pool, user_prefs, rng=rng, n_runs=3, max_steps=2000
    )
    print(f"\nBest score: {score:.4f}")
    print(f"Best plan slots: {plan.slots}")
