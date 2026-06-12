"""simulated annealing for weekly meal plan optimization."""
from __future__ import annotations

import math
import random

from src.meal_plan import MealPlan
from src.evaluation.objective import score_plan

### General Citation: Used Gen AI to document, organize, and debug
### AI CITATION: Used Gen AI to debug hard penalty constraint - had issue with getting
### all algorithms to escape plateau of hard penalty
def simulated_annealing(
    pool,
    user_prefs,
    rng: random.Random | None = None,
    max_steps: int = 10_000,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.995,
    min_temp: float = 1e-4,
):
    """
    single simulated annealing run from a random start.

    at each step, sample one neighbor (e.g. single meal swap). always accept
    improvements; accept worse moves with probability exp(delta / T),
    where delta is negative. temperature cools at each step.

    The 'best seen so far' plan is tracked separately from the current plan,
    so accepting a worse move late in the schedule doesn't lose progress.

    returns (best_plan, best_score).
    """
    if rng is None:
        rng = random.Random()

    current = MealPlan.random(pool_size=len(pool), rng=rng)
    current_score = score_plan(current, pool, user_prefs)

    best_plan = current
    best_score = current_score

    temp = initial_temp
    for _ in range(max_steps):
        # once we reach min temp, more steps won't do anything useful so exit
        if temp < min_temp:
            break
        
        # take one candidate (meal swap) and check whether it is better or worse
        neighbor = current.neighbor(pool_size=len(pool), rng=rng)
        neighbor_score = score_plan(neighbor, pool, user_prefs)
        delta = neighbor_score - current_score

        # improvements accepted unconditionally + short circuits
        # only check delta / temp when delta is 0 or negative
        if delta > 0 or rng.random() < math.exp(delta / temp):
            current = neighbor
            current_score = neighbor_score
            # update the best so far tracker if this score is better
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
    """
    run simulated_annealing n_runs times and return the best overall.

    rng threading for evaluation plan - seeded random.Random through every
    random choice means given seed reproduces exact same run - good for comparison
    of algorithms.

    returns (best_plan, best_score).
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
        print(f"run {i + 1}/{n_runs}: score = {score:.4f}")
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
    print(f"\nbest score: {score:.4f}")
    print(f"best plan slots: {plan.slots}")
