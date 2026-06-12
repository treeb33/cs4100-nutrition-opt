"""recipe representation and dataset loading for the meal planner."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Recipe:
    id: int
    name: str
    minutes: int          # total prep + cook time
    calories: float       # kcal (absolute)
    fat_pdv: float        # everything below is % daily value, not grams
    sugar_pdv: float
    sodium_pdv: float
    protein_pdv: float
    sat_fat_pdv: float
    carbs_pdv: float
    ingredients: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def has_tag(self, tag: str) -> bool:
        return tag.lower() in self.tags


def _parse_list(raw: str) -> tuple[str, ...]:
    """Food.com stores list columns (ingredients, tags) as stringified lists."""
    try:
        return tuple(str(s).strip().lower() for s in ast.literal_eval(raw))
    except (ValueError, SyntaxError):
        return ()

# added time constraint to reduce dataset to ~100K recipes
def load_recipes(
    csv_path: str | Path,
    min_minutes: int | None = 5,
    max_minutes: int | None = 35,
) -> list[Recipe]:
    """Load recipes from the Food.com RAW_recipes.csv export.

    The `nutrition` column is a stringified 7-element list:
        [calories(kcal), total_fat(PDV), sugar(PDV), sodium(PDV),
         protein(PDV), saturated_fat(PDV), carbohydrates(PDV)]
    Only calories is an absolute value; the rest are % daily value. The
    objective function should either work in PDV consistently or convert
    to grams up front -- flagging this so it doesn't silently bite us.

    Filters (defaults chosen to cut the raw ~231k down to ~108k):
      - `min_minutes` drops zero-minute data-error rows and assembly-only
        "recipes" that don't represent real meals.
      - `max_minutes` caps cook time so the search space stays realistic
        for weekly meal planning; 35 keeps recipes that fit a weeknight.

    Pass `None` for either filter to disable it (useful for inspecting
    the full raw distribution).
    """
    df = pd.read_csv(csv_path)
    recipes: list[Recipe] = []
    for row in df.itertuples(index=False):
        try:
            nutrition = ast.literal_eval(row.nutrition)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(nutrition, (list, tuple)) or len(nutrition) != 7:
            continue

        recipe = Recipe(
            id=int(row.id),
            name=str(row.name),
            minutes=int(row.minutes),
            calories=float(nutrition[0]),
            fat_pdv=float(nutrition[1]),
            sugar_pdv=float(nutrition[2]),
            sodium_pdv=float(nutrition[3]),
            protein_pdv=float(nutrition[4]),
            sat_fat_pdv=float(nutrition[5]),
            carbs_pdv=float(nutrition[6]),
            ingredients=_parse_list(row.ingredients),
            tags=_parse_list(row.tags),
        )
        if min_minutes is not None and recipe.minutes < min_minutes:
            continue
        if max_minutes is not None and recipe.minutes > max_minutes:
            continue
        recipes.append(recipe)
    return recipes


if __name__ == "__main__":
    # Smoke test: python recipe.py ../data/RAW_recipes.csv
    import sys

    recipes = load_recipes(sys.argv[1])
    print(f"Loaded {len(recipes)} recipes")
    print(recipes[0])