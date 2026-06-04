"""Hill climbing with random restarts for weekly meal plan optimization."""

import random
from src.meal_plan import MealPlan
from src.evaluation.objective import score_plan


def hill_climb(pool, user_prefs, rng=None, max_steps=1000):
    """Single hill climbing run from a random start.
    
    At each step, samples one neighbor (single meal swap) and keeps it
    if it improves the score. Stops when no improvement or max_steps hit.
    
    Returns (best_plan, best_score).
    """
    if rng is None:
        rng = random.Random()

    current = MealPlan.random(pool_size=len(pool), rng=rng)
    current_score = score_plan(current, pool, user_prefs)

    for _ in range(max_steps):
        neighbor = current.neighbor(pool_size=len(pool), rng=rng)
        neighbor_score = score_plan(neighbor, pool, user_prefs)

        if neighbor_score > current_score:
            current = neighbor
            current_score = neighbor_score

    return current, current_score


def hill_climb_with_restarts(pool, user_prefs, rng=None, n_restarts=10, max_steps=1000):
    """Run hill_climb n_restarts times, return the best result overall.
    
    This is the main entry point for experiments.
    
    Returns (best_plan, best_score).
    """
    if rng is None:
        rng = random.Random()

    best_plan = None
    best_score = float("-inf")

    for i in range(n_restarts):
        plan, score = hill_climb(pool, user_prefs, rng=rng, max_steps=max_steps)
        print(f"  Restart {i+1}/{n_restarts}: score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_plan = plan

    return best_plan, best_score


if __name__ == "__main__":
    # Smoke test with dummy scorer (objective.py is still a stub)
    from src.recipe import load_recipes

    # Mock user prefs
    user_prefs = {
        "calorie_target": 2000,
        "budget": 100,
        "allergens": [],
    }

    rng = random.Random(42)

    # Run without real data using a fake pool
    class FakeRecipe:
        def __init__(self, id): self.id = id

    fake_pool = [FakeRecipe(i) for i in range(300)]
    plan, score = hill_climb_with_restarts(fake_pool, user_prefs, rng=rng, n_restarts=3, max_steps=100)
    print(f"\nBest plan slots: {plan.slots[:7]}...")
    print(f"Best score: {score}")