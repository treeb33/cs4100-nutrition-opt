"""Tests for the scoring function and the graded hard penalty."""
import random

from src.evaluation.objective import HARD_PENALTY, is_feasible, score_plan
from src.meal_plan import NUM_SLOTS, MealPlan
from src.recipe import Recipe

### AI CITATION: Used Gen AI to write the necessary tests I believed
### were required for adequate code coverage
def _make_recipe(
    rid: int,
    *,
    calories: float = 500.0,
    protein_pdv: float = 30.0,
    fat_pdv: float = 25.0,
    carbs_pdv: float = 40.0,
    sodium_pdv: float = 20.0,
    sugar_pdv: float = 30.0,
    sat_fat_pdv: float = 10.0,
    minutes: int = 30,
    ingredients: tuple[str, ...] = ("chicken", "rice", "broccoli"),
    tags: tuple[str, ...] = (),
):
    return Recipe(
        id=rid,
        name=f"r{rid}",
        minutes=minutes,
        calories=calories,
        fat_pdv=fat_pdv,
        sugar_pdv=sugar_pdv,
        sodium_pdv=sodium_pdv,
        protein_pdv=protein_pdv,
        sat_fat_pdv=sat_fat_pdv,
        carbs_pdv=carbs_pdv,
        ingredients=ingredients,
        tags=tags,
    )


def _uniform_plan(pool_index: int) -> MealPlan:
    """A plan where all 21 slots point at the same pool recipe."""
    return MealPlan(slots=[pool_index] * NUM_SLOTS)


def test_feasible_plan_scores_above_negative_hard_penalty():
    # 3 of these recipes per day = 1500 kcal -- inside [1000, 4000]
    pool = [_make_recipe(0, calories=500, sodium_pdv=20)]
    plan = _uniform_plan(0)
    score = score_plan(plan, pool, {"allergens": []})
    assert score > -HARD_PENALTY
    assert is_feasible(plan, pool, {"allergens": []})


def test_allergen_kicks_plan_into_hard_penalty():
    safe = _make_recipe(0, ingredients=("chicken", "rice"))
    nuts = _make_recipe(1, ingredients=("peanut butter", "bread"))
    pool = [safe, nuts]
    plan = MealPlan(slots=[1] + [0] * (NUM_SLOTS - 1))   # one allergen recipe
    score = score_plan(plan, pool, {"allergens": ["peanut"]})
    assert score < -HARD_PENALTY
    assert not is_feasible(plan, pool, {"allergens": ["peanut"]})


def test_calorie_range_violation_triggers_hard_penalty():
    # 3 x 100 kcal = 300 / day, well below 1000
    pool = [_make_recipe(0, calories=100, sodium_pdv=10)]
    plan = _uniform_plan(0)
    assert score_plan(plan, pool, {"allergens": []}) < -HARD_PENALTY
    assert not is_feasible(plan, pool, {"allergens": []})

    # 3 x 2000 kcal = 6000 / day, well above 4000
    pool_big = [_make_recipe(0, calories=2000, sodium_pdv=10)]
    plan_big = _uniform_plan(0)
    assert score_plan(plan_big, pool_big, {"allergens": []}) < -HARD_PENALTY


def test_graded_penalty_orders_infeasible_plans():
    """Worse calorie violations score worse than mild ones.

    This is what was missing before: a flat -HARD_PENALTY meant local
    search had no gradient out of the infeasible region.
    """
    # daily total = 3 * per-recipe calories; min feasible day is 1000 kcal
    pool_mild = [_make_recipe(0, calories=300, sodium_pdv=10)]   # 900/day, 100 short
    pool_worse = [_make_recipe(0, calories=100, sodium_pdv=10)]  # 300/day, 700 short

    s_mild = score_plan(_uniform_plan(0), pool_mild, {"allergens": []})
    s_worse = score_plan(_uniform_plan(0), pool_worse, {"allergens": []})

    assert s_mild < -HARD_PENALTY
    assert s_worse < s_mild  # worse violation = lower score


def test_any_feasible_score_beats_any_infeasible_score():
    feasible_pool = [_make_recipe(0, calories=500, sodium_pdv=20)]
    infeasible_pool = [_make_recipe(0, calories=100, sodium_pdv=10)]

    s_feasible = score_plan(_uniform_plan(0), feasible_pool, {"allergens": []})
    s_infeasible = score_plan(_uniform_plan(0), infeasible_pool, {"allergens": []})

    assert s_feasible > s_infeasible


def test_score_plan_returns_float():
    pool = [_make_recipe(0, calories=500)]
    score = score_plan(_uniform_plan(0), pool, {"allergens": []})
    assert isinstance(score, float)


def test_is_feasible_is_consistent_with_score_threshold():
    rng = random.Random(0)
    pool = [
        _make_recipe(i, calories=400 + 50 * i, sodium_pdv=15)
        for i in range(20)
    ]
    for _ in range(30):
        plan = MealPlan.random(pool_size=len(pool), rng=rng)
        score = score_plan(plan, pool, {"allergens": []})
        feasible = is_feasible(plan, pool, {"allergens": []})
        # contract: score > -HARD_PENALTY iff is_feasible is True
        assert (score > -HARD_PENALTY) == feasible
