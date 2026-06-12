
"""
Objective function for evaluating a weekly meal plan.
A plan is a list of 21 recipe IDs (7 days x 3 meals).
Returns a scalar score — higher is better.
Hard constraint violations apply a large penalty.
"""
### AI CITATION: used AI for comments and documentation,
### and for thinking through how to structure and scale the penalty
### weights in the objective function.


from __future__ import annotations
from collections import Counter
from src.meal_plan import MealPlan, DAYS, MEALS_PER_DAY

NUM_SLOTS = DAYS * MEALS_PER_DAY
HARD_PENALTY = 1e6

# default targets and weights shared across all three search algorithms
# NOTE: macros are in percent daily value to match how Recipe stores them, calories is raw kcal
# user preferences with defaults where targets are values we optimize toward,
# limits are hard caps, and weights control how much each factor affects the overall score
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
    # weights per group
    "w_calories":        2.0,
    "w_protein":         1.5,
    "w_fat":             1.0,
    "w_carbs":           1.0,
    "w_variety":         1.0,
    "w_ingredient_reuse":0.5,
    "w_prep_time":       0.5,
    "w_sugar":           0.8,
}


# given a day, returns the 3 recipes
# plan.day(d) gives the slot indicies, pool[i] looks up the actual recipe
def _day_recipes(plan: MealPlan, pool: list, d: int) -> list:
    return [pool[i] for i in plan.day(d)]


def _hard_violation_severity(plan: MealPlan, pool: list, prefs: dict, recipes: list) -> float:
    """
    return 0.0 if the plan is feasible, otherwise a positive scalar
    proportional to how badly the hard constraints are violated.

    A flat hard penalty makes ~99.8% of random starts identical from the
    search's point of view, so SA and hill climbing have no gradient out
    of the infeasible region. Grading the penalty by severity gives the
    search a smooth signal that points toward feasibility
    """
    # start at 0 - if nothing is violated we return 0 and the plan is feasible
    severity = 0.0

    # allergen check
    allergens = [a.lower().strip() for a in prefs["allergens"]]
    if allergens:
        for r in recipes:
            # if any allergen appears in any ingredient in this recipe then add a penalty
            if any(a in ing for a in allergens for ing in r.ingredients):
                severity += 1.0   # one full unit per allergen-containing recipe

    # check each day's total calories
    for d in range(DAYS):
        kcal = sum(r.calories for r in _day_recipes(plan, pool, d))
        if kcal < 1000:
            # if too few calories, then penalize proportinally to how far under the limit we are
            severity += (1000 - kcal) / 1000
        elif kcal > 4000:
            # too may calories - penalize proportionally to how far over the limit we are
            severity += (kcal - 4000) / 1000

    # check average daily sodium across the whole week
    avg_sodium = sum(r.sodium_pdv for r in recipes) / DAYS
    if avg_sodium > prefs["sodium_limit"]:
        # again, penalize prop to how much over the limit we are
        severity += (avg_sodium - prefs["sodium_limit"]) / prefs["sodium_limit"]

    return severity


def _soft_score(plan: MealPlan, pool: list, prefs: dict, recipes: list) -> float:
    # start at 0 and subtract penalities and add rewards as we iterae
    score = 0.0

    # check each day's calories against the target
    for d in range(DAYS):
        kcal = sum(r.calories for r in _day_recipes(plan, pool, d))

        # how far outside the tolerance are we, if within 200 of the target then no penalty
        overage = abs(kcal - prefs["calorie_target"]) - prefs["calorie_tolerance"]
        if overage > 0:
            # penalize scale by the calorie weight
            score -= prefs["w_calories"] * overage / prefs["calorie_target"]

    # defining three macros we care about - protein, fat, carbs
    # tuple - (pref key, recipe attribute, weight)
    macros = [
        ("protein_target", "protein_pdv", prefs["w_protein"]),
        ("fat_target",     "fat_pdv",     prefs["w_fat"]),
        ("carbs_target",   "carbs_pdv",   prefs["w_carbs"]),
    ]

    # for each day check if we are hitting all three macro minimums
    for d in range(DAYS):
        day = _day_recipes(plan, pool, d)
        for target_key, attr, weight in macros:
            # how off from hitting the target we are (0 if we are hitting the target)
            shortfall = max(0, prefs[target_key] - sum(getattr(r, attr) for r in day))
            # penalty
            score -= weight * (shortfall / prefs[target_key])

    # sugar penalty - only kicks in if we are consistently over the weekly average
    avg_sugar = sum(r.sugar_pdv for r in recipes) / DAYS
    if avg_sugar > prefs["sugar_limit"]:
        score -= prefs["w_sugar"] * (avg_sugar - prefs["sugar_limit"]) / prefs["sugar_limit"]

    # variety penalty - count how many recipes appear more than once in the week
    #e.g. if you eat the same recipe 3 times that's two repeats
    repeats = sum(max(0, c - 1) for c in Counter(plan.slots).values())
    score -= prefs["w_variety"] * (repeats / NUM_SLOTS)

    # ingredient reuse reward: if the same ingredient appears across multiple meals
    # that means a simpler shopping list, so we reward it
    all_ings = [ing for r in recipes for ing in r.ingredients]
    reused = sum(1 for c in Counter(all_ings).values() if c > 1)
    score += prefs["w_ingredient_reuse"] * (reused / max(len(set(all_ings)), 1))

    # penalize anything that takes too long to make
    time_penalty = sum(
        (r.minutes - prefs["max_minutes"]) / prefs["max_minutes"]
        for r in recipes if r.minutes > prefs["max_minutes"]
    )
    score -= prefs["w_prep_time"] * (time_penalty / NUM_SLOTS)

    return score



def score_plan(plan: MealPlan, pool: list, user_prefs: dict) -> float:
    """
    plan: MealPlan with 21 slots (indices into pool)
    pool: list of Recipe objects
    user_prefs: dict with keys like calorie_target, allergens, etc.
    Returns a scalar score — higher is better.

    Infeasible plans score -HARD_PENALTY minus a graded severity term, so
    the search still has a gradient pointing toward the feasible region.
    Any feasible plan scores strictly better than any infeasible one.
    """
    # user preferences override the default prefs that we set
    prefs = {**DEFAULT_PREFS, **user_prefs}
    # actual recipe objects for all 21 slots
    recipes = plan.recipes(pool)

    # first we check the HARD CONSTRAINTS
    # these are constraints that cannot be failed, these must pass before
    # checking soft goals. if any hard constraint is violeted, then return
    # a large negative score. severity term means worse violations get worse scores
    severity = _hard_violation_severity(plan, pool, prefs, recipes)
    if severity > 0:
        return -HARD_PENALTY - severity

    # if all hard constraints are satisfied score the soft goals
    return _soft_score(plan, pool, prefs, recipes)


# checks whether a plan passes all the hard constraints
# returns True if the plan is feasible (meets hard constraints), False if not
def is_feasible(plan: MealPlan, pool: list, user_prefs: dict) -> bool:
    """
    true iff the plan satisfies every hard constraint.

    used by the experiments harness to measure each algorithm's
    constraint-satisfaction rate across seeds.
    """
    prefs = {**DEFAULT_PREFS, **user_prefs}
    recipes = plan.recipes(pool)
    return _hard_violation_severity(plan, pool, prefs, recipes) == 0