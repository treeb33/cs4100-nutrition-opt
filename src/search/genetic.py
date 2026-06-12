"""genetic algo for weekly meal plan optimizer"""
# AI CITATION: used Gen AI for comments and to help think through
# how single point crossover should split the 21 meal slots between two parents


import random
from src.recipe import load_recipes
from src.meal_plan import MealPlan, NUM_SLOTS
from src.evaluation.objective import score_plan


# only breed 80% of the time, otherwise pass parents through unchanged
# this keeps some diversity in the population
CROSSOVER_RATE = 0.8


def _pick_parent(pop, fitness, k=3, rng=None):
    # randomly sample k plans and return the best scoring one
    # using k=3 instead of always pikcing the top plan keeps the populaton diverse
    # weaker plans still have chance to be parents which avoids fast convergence
    rng = rng or random.Random()
    entries = rng.sample(range(len(pop)), k)
    winner = max(entries, key=lambda i: fitness[i])
    return pop[winner]


def _combine(plan_a, plan_b, rng=None):
    # SINGLE POINT CROSSOVER
    # combines traits of two parent solutions to create two children
    # pick a random point to swap the points of two parents
    # e.g. if cut = 10, child_a gets slots 0-9 from plan a and slots 10-20 from
    # plan_b
    # skip crossover 20% of the time and return parents unchanged
    rng = rng or random.Random()
    if rng.random() > CROSSOVER_RATE:
        return plan_a, plan_b
    cut = rng.randint(1, NUM_SLOTS - 1)
    child_a = MealPlan(slots=plan_a.slots[:cut] + plan_b.slots[cut:])
    child_b = MealPlan(slots=plan_b.slots[:cut] + plan_a.slots[cut:])
    return child_a, child_b


def _apply_mutation(plan, pool_size, rate=0.1, rng=None):
    # go through every slot and randomly replace it with probability=rate
    # this introduces new recipes that weren't in the population before
    # without mutation, crossover can only remix what already exists and  we'd get stuck
    rng = rng or random.Random()
    updated = list(plan.slots)
    for i in range(NUM_SLOTS):
        if rng.random() < rate:
            updated[i] = rng.randrange(pool_size)
    return MealPlan(slots=updated)


def run_genetic(pool, user_prefs, rng=None, pop_size=50, n_generations=100, mutation_rate=0.1):
    if rng is None:
        rng = random.Random()

    pool_size = len(pool)

    # start with a completely random population so we explore different parts of the space
    print(f"initializing population of {pop_size} random plans:")
    pop = [MealPlan.random(pool_size=pool_size, rng=rng) for _ in range(pop_size)]

    best_plan = None
    best_score = float("-inf")
    no_improve_streak = 0

    print(f"starting evolution over {n_generations} generations\n")

    for gen in range(n_generations):
        # score everyone in the curr pop using the objective function
        fitness = [score_plan(p, pool, user_prefs) for p in pop]

        avg_score = sum(fitness) / len(fitness)
        top_idx = max(range(pop_size), key=lambda i: fitness[i])
        gen_best = fitness[top_idx]

        # update global best if this generation found something better
        # we can track this seperately because the pop can get worse between generations
        if gen_best > best_score:
            improvement = gen_best - best_score
            best_score = gen_best
            best_plan = pop[top_idx]
            no_improve_streak = 0
            improved_str = f"  ↑ improved by {improvement:.4f}"
        else:
            no_improve_streak += 1
            improved_str = f"  (no improvement for {no_improve_streak} generations)"

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"  gen {gen+1:>3}/{n_generations} | best: {gen_best:.4f} | avg: {avg_score:.4f} |{improved_str}")

        # always take the best plan into the next gen unchanged
        # without this we could potentially lose the best solution
        next_gen = [pop[top_idx]]

        # breed children until we refill the population back to the pop size
        while len(next_gen) < pop_size:
            # pick two parents via tournament selection
            mom = _pick_parent(pop, fitness, rng=rng)
            dad = _pick_parent(pop, fitness, rng=rng)
            # combine them to make two children
            child_a, child_b = _combine(mom, dad, rng=rng)
            # mutate both children to introduce new recipes
            child_a = _apply_mutation(child_a, pool_size, rate=mutation_rate, rng=rng)
            child_b = _apply_mutation(child_b, pool_size, rate=mutation_rate, rng=rng)
            next_gen.extend([child_a, child_b])

        # trim back to pop_size in case we went over the size
        pop = next_gen[:pop_size]

    print(f"\ndone. Best score across all generations: {best_score:.4f}")
    return best_plan, best_score


if __name__ == "__main__":

    pool = load_recipes("data/RAW_recipes.csv")
    print(f"loaded {len(pool)} recipes\n")

    user_prefs = {"calorie_target": 2000, "allergens": []}
    rng = random.Random(42)

    plan, score = run_genetic(pool, user_prefs, rng=rng, pop_size=30, n_generations=50)
    print(f"\nbest plan slots: {plan.slots[:7]}...")
    print(f"final score: {score:.4f}")
