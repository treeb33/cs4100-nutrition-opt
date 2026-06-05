"""
Objective function for evaluating a weekly meal plan.
A plan is a list of 21 recipe IDs (7 days x 3 meals).
Returns a scalar score — higher is better.
Hard constraint violations apply a large penalty.
"""

from __future__ import annotations
from collections import Counter
from src.meal_plan import MealPlan, DAYS, MEALS_PER_DAY

NUM_SLOTS = DAYS * MEALS_PER_DAY
HARD_PENALTY = 1e6

# all the default targets and weights we're using across all three algorithms
# keeping macros in PDV to stay consistent with how Recipe stores them
# calories is the only one stored as actual kcal (not PDV)
DEFAULT_PREFS = {
    "calorie_target":    2000,
    "calorie_tolerance": 200,
    "protein_target":    50,
    "fat_target":        65,
    "carbs_target":      130,
    "sodium_limit":      100,
    "sugar_limit":       200,
    "max_minutes":       60,
    "allergens":         [],
    "w_calories":        2.0,
    "w_protein":         1.5,
    "w_fat":             1.0,
    "w_carbs":           1.0,
    "w_variety":         1.0,
    "w_ingredient_reuse":0.5,
    "w_prep_time":       0.5,
    "w_sugar":           0.8,
}


def _day_recipes(plan: MealPlan, pool: list, d: int) -> list:
    # just a helper so we're not repeating this lookup everywhere
    return [pool[i] for i in plan.day(d)]


def score_plan(plan: MealPlan, pool: list, user_prefs: dict) -> float:
    """
    plan: MealPlan with 21 slots (indices into pool)
    pool: list of Recipe objects
    user_prefs: dict with keys like calorie_target, allergens, etc.
    Returns a scalar score — higher is better.
    Hard constraint violations apply a large negative penalty.
    """
    prefs = {**DEFAULT_PREFS, **user_prefs}
    recipes = plan.recipes(pool)
    score = 0.0

    # KEY CONSTRAITNTS
    # these return immediately so the search doesn't waste time on bad plans
    # if any recipe has a blacklisted ingredient, immediately throw it out
    allergens = [a.lower().strip() for a in prefs["allergens"]]
    if allergens:
        for r in recipes:
            if any(a in ing for a in allergens for ing in r.ingredients):
                return -HARD_PENALTY

    # each day has to be within a realistic calorie range
    for d in range(DAYS):
        kcal = sum(r.calories for r in _day_recipes(plan, pool, d))
        if not (1000 <= kcal <= 4000):
            return -HARD_PENALTY

    # hard cap on sodium
    if sum(r.sodium_pdv for r in recipes) / DAYS > prefs["sodium_limit"]:
        return -HARD_PENALTY

    # penalize days that fall outside the calorie tolerance band
    for d in range(DAYS):
        kcal = sum(r.calories for r in _day_recipes(plan, pool, d))
        overage = abs(kcal - prefs["calorie_target"]) - prefs["calorie_tolerance"]
        if overage > 0:
            score -= prefs["w_calories"] * overage / prefs["calorie_target"]

    # penalize missing macro minimums
    for d in range(DAYS):
        day = _day_recipes(plan, pool, d)
        for key, attr, weight in [
            ("protein_target", "protein_pdv", prefs["w_protein"]),
            ("fat_target",     "fat_pdv",     prefs["w_fat"]),
            ("carbs_target",   "carbs_pdv",   prefs["w_carbs"]),
        ]:
            miss = max(0, prefs[key] - sum(getattr(r, attr) for r in day)) / prefs[key]
            score -= weight * miss

    # soft penalty for too much sugar, only kicks in if we're consistently over
    daily_sugar = sum(r.sugar_pdv for r in recipes) / DAYS
    if daily_sugar > prefs["sugar_limit"]:
        score -= prefs["w_sugar"] * (daily_sugar - prefs["sugar_limit"]) / prefs["sugar_limit"]

    # penalize repeated recipes, eating the same thing 3x a week is not a meal plan
    repeats = sum(max(0, c - 1) for c in Counter(plan.slots).values())
    score -= prefs["w_variety"] * (repeats / NUM_SLOTS)

    # reward ingredient overlap across meals : less waste, more realistic shopping
    all_ingredients = [ing for r in recipes for ing in r.ingredients]
    reused = sum(1 for c in Counter(all_ingredients).values() if c > 1)
    score += prefs["w_ingredient_reuse"] * (reused / max(len(set(all_ingredients)), 1))

    # penalize meals that take too long to make
    time_penalty = sum(
        (r.minutes - prefs["max_minutes"]) / prefs["max_minutes"]
        for r in recipes if r.minutes > prefs["max_minutes"]
    )
    score -= prefs["w_prep_time"] * (time_penalty / NUM_SLOTS)

    return score
