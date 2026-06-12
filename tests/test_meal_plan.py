"""Tests for the MealPlan state representation and neighbor move."""
import random

from src.meal_plan import DAYS, MEALS_PER_DAY, NUM_SLOTS, MealPlan

### AI CITATION: Used Gen AI to write the necessary tests I believed
### were required for adequate code coverage
def test_random_plan_fills_21_slots():
    plan = MealPlan.random(pool_size=100, rng=random.Random(0))
    assert len(plan.slots) == NUM_SLOTS == 21


def test_random_plan_uses_only_pool_indices():
    pool_size = 50
    plan = MealPlan.random(pool_size=pool_size, rng=random.Random(0))
    assert all(0 <= s < pool_size for s in plan.slots)


def test_neighbor_differs_by_exactly_one_slot():
    rng = random.Random(7)
    plan = MealPlan.random(pool_size=100, rng=rng)
    # the move can sample the same recipe back; loop until it actually changes
    # so the test reflects the intended invariant (at most one slot differs;
    # in expectation, exactly one)
    for _ in range(50):
        neighbor = plan.neighbor(pool_size=100, rng=rng)
        diffs = [i for i in range(NUM_SLOTS) if plan.slots[i] != neighbor.slots[i]]
        assert len(diffs) <= 1
        if diffs:
            return
    # extremely unlikely to hit 50 no-ops with pool_size=100
    raise AssertionError("neighbor never changed any slot in 50 tries")


def test_neighbor_returns_new_object():
    rng = random.Random(0)
    plan = MealPlan.random(pool_size=10, rng=rng)
    neighbor = plan.neighbor(pool_size=10, rng=rng)
    assert neighbor is not plan
    assert neighbor.slots is not plan.slots


def test_day_returns_three_slot_indices():
    plan = MealPlan.random(pool_size=100, rng=random.Random(0))
    for d in range(DAYS):
        day = plan.day(d)
        assert len(day) == MEALS_PER_DAY == 3


def test_seeded_rng_is_reproducible():
    plan_a = MealPlan.random(pool_size=200, rng=random.Random(42))
    plan_b = MealPlan.random(pool_size=200, rng=random.Random(42))
    assert plan_a.slots == plan_b.slots
