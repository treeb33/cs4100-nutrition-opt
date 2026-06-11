"""Simulated annealing for weekly meal plan optimization."""

import random
import math
from src.meal_plan import MealPlan
from src.evaluation.objective import score_plan


def simulated_annealing(pool, user_prefs, rng=None, max_steps=5000, temp_init=10.0, temp_min=0.1):
    """One Simmulated Annealing run from a random starting plan

    Takes worse solutions sometimes so we don't get stuck in local optima.
    Temperature drops exponentially each step until we hit temp_min.

    Returns (best_plan, best_score).
    """
    if rng is None:
        rng = random.Random()

    # start from a random plan and score it
    plan = MealPlan.random(pool_size=len(pool), rng=rng)
    plan_score = score_plan(plan, pool, user_prefs)

    # keep track of the best we've seen since current can get worse
    best_plan = plan
    best_score = plan_score

    # precompute how much to multiply temp each step so we hit temp_min at the end
    decay = (temp_min / temp_init) ** (1 / max_steps)
    temp = temp_init

    for _ in range(max_steps):
        # swap one meal slot and see if it's better
        candidate = plan.neighbor(pool_size=len(pool), rng=rng)
        candidate_score = score_plan(candidate, pool, user_prefs)

        diff = candidate_score - plan_score

        # always take improvements, but also take worse solutions with
        # probability e^(diff/temp) — this lets us escape local optima
        # early on when temp is high, less so as it cools down
        if diff > 0 or rng.random() < math.exp(diff / temp):
            plan = candidate
            plan_score = candidate_score

        # update best separately since we might have moved to a worse plan
        if plan_score > best_score:
            best_score = plan_score
            best_plan = plan

        temp *= decay

    return best_plan, best_score


def simulated_annealing_with_restarts(pool, user_prefs, rng=None, n_restarts=5, max_steps=5000, temp_init=10.0, temp_min=0.1):
    """Run SA n_restarts times and return the best plan we found.

    This is what the experiments script calls.

    Returns (best_plan, best_score).
    """
    if rng is None:
        rng = random.Random()

    best_plan = None
    best_score = float("-inf")

    for i in range(n_restarts):
        # each restart gets a fresh random plan
        plan, score = simulated_annealing(pool, user_prefs, rng=rng, max_steps=max_steps, temp_init=temp_init, temp_min=temp_min)
        print(f"  Restart {i+1}/{n_restarts}: score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_plan = plan

    return best_plan, best_score


if __name__ == "__main__":
    from src.recipe import load_recipes

    pool = load_recipes("data/RAW_recipes.csv")
    user_prefs = {"calorie_target": 2000, "allergens": []}
    rng = random.Random(42)

    plan, score = simulated_annealing_with_restarts(pool, user_prefs, rng=rng, n_restarts=2, max_steps=500)
    print(f"\nBest plan slots: {plan.slots[:7]}...")
    print(f"Best score: {score}")