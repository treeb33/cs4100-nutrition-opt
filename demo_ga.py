"""
Demo script — run genetic algorithm and print a readable weekly meal plan.

Usage (from repo root, with venv active):
    python demo_ga.py --csv data/RAW_recipes.csv
"""

import argparse
import random

from src.recipe import load_recipes, curate_pool
from src.evaluation.objective import DEFAULT_PREFS
from src.search.genetic import run_genetic

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MEALS = ["Breakfast", "Lunch", "Dinner"]


def print_plan(plan, pool, score):
    print("\n" + "=" * 55)
    print("  YOUR WEEKLY MEAL PLAN  (Genetic Algorithm)")
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
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=100)
    args = parser.parse_args()

    print(f"\nLoading recipes from {args.csv} ...")
    raw = load_recipes(args.csv)
    pool = curate_pool(raw, target_size=args.pool_size, rng=random.Random(args.seed))
    print(f"Curated pool: {len(pool)} recipes")

    user_prefs = dict(DEFAULT_PREFS)

    print(f"\nRunning genetic algorithm (seed={args.seed}, pop={args.pop_size}, generations={args.generations}) ...\n")
    rng = random.Random(args.seed)
    plan, score = run_genetic(
        pool, user_prefs, rng=rng,
        pop_size=args.pop_size,
        n_generations=args.generations
    )

    print_plan(plan, pool, score)


if __name__ == "__main__":
    main()