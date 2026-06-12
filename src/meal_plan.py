"""weekly meal plan state and neighbor generation for local search."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recipe import Recipe

DAYS = 7
MEALS_PER_DAY = 3                  # breakfast, lunch, dinner
NUM_SLOTS = DAYS * MEALS_PER_DAY   # 21

### General Citation: Used Gen AI to document, organize, and debug
@dataclass
class MealPlan:
    """a full week as one state: 21 recipe slots.

    slots store indices into a shared recipe pool (not Recipe objects), so
    states are easier to compare. 
    """
    slots: list[int]   # length 21; each entry is an index into the recipe pool

    @classmethod
    def random(cls, pool_size: int, rng: random.Random | None = None) -> "MealPlan":
        r = rng or random
        return cls(slots=[r.randrange(pool_size) for _ in range(NUM_SLOTS)])

    def neighbor(self, pool_size: int, rng: random.Random | None = None) -> "MealPlan":
        """return a NEW plan differing by exactly one meal.

        single-swap move is the shared primitive:
          - hill climbing scans these and keeps the best improving one
          - simulated annealing samples one and accepts according to probability
          - the genetic algorithm uses it as its mutation operator
        """
        r = rng or random
        new_slots = list(self.slots)
        i = r.randrange(NUM_SLOTS)
        new_slots[i] = r.randrange(pool_size)
        return MealPlan(slots=new_slots)

    def recipes(self, pool: list["Recipe"]) -> list["Recipe"]:
        return [pool[i] for i in self.slots]

    def day(self, d: int) -> list[int]:
        """Return the three slot indices for day d (0-6)."""
        start = d * MEALS_PER_DAY
        return self.slots[start:start + MEALS_PER_DAY]


if __name__ == "__main__":
    # reproducibility check: a seeded RNG gives identical plans,
    # matters for the "run each algorithm N times" evaluation plan.
    rng = random.Random(42)
    plan = MealPlan.random(pool_size=1000, rng=rng)
    print("slots:", plan.slots)
    print("neighbor:", plan.neighbor(pool_size=1000, rng=rng).slots)