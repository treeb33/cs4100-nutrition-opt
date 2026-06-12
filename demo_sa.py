## ## AI assistance (Claude) was used to help structure the demo script
## and format the weekly meal plan output for readability.

"""
Demo script — run simulated annealing and print a readable weekly meal plan.

Usage (from repo root, with venv active):
    python demo_sa.py --csv data/RAW_recipes.csv
"""

import argparse
import random

from src.recipe import load_recipes, curate_pool
from src.evaluation.objective import DEFAULT_PREFS, score_plan
from src.search.simulated_annealing import simulated_annealing_multi

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MEALS = ["Breakfast", "Lunch", "Dinner"]


def print_plan(plan, pool, score):
    print("\n" + "=" * 55)
    print("  YOUR WEEKLY MEAL PLAN  (Simulated Annealing)")
    print("=" * 55)

    for d, day_name in enumerate(DAYS):
        print(f"\n{day_name}")
        day_slots = plan.day(d)
        day_calories = 0
        for m, meal_name in enumerate(MEALS):
            recipe = pool[day_slots[m]]
            day_calories += recipe.calories
            print(f"  {meal_name:<12} {recipe.name[:30]:<32} {round(recipe.calories)} cal")
        print(f"  {'Daily total':<44} {round(day_calories)} cal")

    print("\n" + "=" * 55)
    print(f"  Final score: {score:.4f}  (higher / closer to 0 is better)")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/RAW_recipes.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=400)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    print(f"\nLoading recipes from {args.csv} ...")
    raw = load_recipes(args.csv)
    pool = curate_pool(raw, target_size=args.pool_size, rng=random.Random(args.seed))
    print(f"Curated pool: {len(pool)} recipes")

    user_prefs = dict(DEFAULT_PREFS)

    print(f"\nRunning simulated annealing (seed={args.seed}, {args.runs} runs) ...\n")
    rng = random.Random(args.seed)
    plan, score = simulated_annealing_multi(
        pool, user_prefs, rng=rng, n_runs=args.runs
    )

    print_plan(plan, pool, score)


if __name__ == "__main__":
    main()