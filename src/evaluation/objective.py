"""
Objective function for evaluating a weekly meal plan.
A plan is a list of 21 recipe IDs (7 days x 3 meals).
Returns a scalar score — higher is better.
Hard constraint violations apply a large penalty.
"""

HARD_PENALTY = 1e6

def score_plan(plan, recipes, user_prefs):
    """
    plan: list of 21 recipe IDs
    recipes: dict mapping recipe_id -> recipe dict
    user_prefs: dict with keys like calorie_target, budget, allergens, etc.
    """
    score = 0.0

    # --- Hard constraints ---
    # (return large negative penalty if violated)

    # --- Soft goals (to be implemented) ---
    # macro targets
    # ingredient reuse / waste reduction
    # variety (penalize repeats)
    # prep time feasibility

    return score