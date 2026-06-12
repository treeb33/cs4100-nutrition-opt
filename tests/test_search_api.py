"""Contract tests: every search algorithm must accept (pool, prefs, rng=...).

The comparison harness drives all three through one call shape; these tests
guard that contract so a future signature change doesn't silently break it.
"""
import random

from src.meal_plan import NUM_SLOTS, MealPlan
from src.recipe import Recipe
from src.search.genetic import run_genetic
from src.search.hill_climbing import hill_climb_with_restarts
from src.search.simulated_annealing import simulated_annealing_multi

### AI CITATION: Used Gen AI to write the necessary tests I believed
### were required for adequate code coverage
def _tiny_pool():
    """A 5-recipe feasible pool so search runs cheaply in CI."""
    return [
        Recipe(
            id=i,
            name=f"r{i}",
            minutes=20,
            calories=500.0,
            fat_pdv=20.0,
            sugar_pdv=15.0,
            sodium_pdv=15.0,
            protein_pdv=25.0,
            sat_fat_pdv=8.0,
            carbs_pdv=35.0,
            ingredients=("chicken", "rice"),
            tags=(),
        )
        for i in range(5)
    ]


def _shared_kwargs():
    return [
        (
            "hill_climbing",
            hill_climb_with_restarts,
            {"n_restarts": 2, "max_steps": 20},
        ),
        (
            "simulated_annealing",
            simulated_annealing_multi,
            {"n_runs": 2, "max_steps": 20},
        ),
        ("genetic", run_genetic, {"pop_size": 6, "n_generations": 5}),
    ]


def test_all_algorithms_accept_pool_prefs_rng():
    pool = _tiny_pool()
    prefs = {"allergens": []}
    for name, algo_fn, kwargs in _shared_kwargs():
        plan, score = algo_fn(pool, prefs, rng=random.Random(0), **kwargs)
        assert isinstance(plan, MealPlan), f"{name} did not return a MealPlan"
        assert isinstance(score, float), f"{name} did not return a float score"
        assert len(plan.slots) == NUM_SLOTS


def test_all_algorithms_are_seed_reproducible():
    """Same seed → same final score for each algorithm.

    Required by the evaluation plan ("run each algorithm N times")
    so the comparison harness produces stable, reportable numbers.
    """
    pool = _tiny_pool()
    prefs = {"allergens": []}
    for name, algo_fn, kwargs in _shared_kwargs():
        _, score_a = algo_fn(pool, prefs, rng=random.Random(123), **kwargs)
        _, score_b = algo_fn(pool, prefs, rng=random.Random(123), **kwargs)
        assert score_a == score_b, f"{name} is not reproducible under a fixed seed"
