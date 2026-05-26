# Meal Planner: Local Search for Weekly Meal Planning

## CS4100 Final Project

**Team Members:** Sanah Menon, Isabella Pozzi, Triya Basu

## Project Overview

Meal planning can be difficult because there are a lot of needs to balance across one week. A person may want to meet their nutrition goals while staying within a grocery budget. They may also have dietary restrictions, limited time to cook, or want to avoid buying ingredients that barely get used.

For this project, we are building a system that generates a weekly meal plan using local search. Rather than picking each meal separately, the system evaluates the full week as one plan. This allows it to search for a plan that fits the user's needs while still being realistic to follow.

## Problem Representation

A single state is one complete weekly meal plan made up of 21 meal slots. Each day includes breakfast, lunch, and dinner, with each slot filled by a recipe from a public recipe dataset.

A neighboring state is created by making a small change to the current plan. For example, the system may replace one meal with a different recipe. This gives the search algorithms a way to gradually improve a plan without trying every possible weekly combination.

## Constraints and Objective Function

The system will score each weekly meal plan based on required constraints and personal preferences.

**Hard constraints:**
- Staying within a daily calorie range
- Avoiding allergens or restricted foods
- Keeping the weekly grocery cost within budget

**Soft goals may include:**
- Staying close to macro targets
- Reusing ingredients to reduce waste
- Avoiding too much repetition across the week
- Keeping the amount of cooking time realistic

A lower score will represent a stronger meal plan. Violations of hard constraints will receive much larger penalties than issues with soft goals. This ensures that the system first focuses on creating a valid plan, then tries to make that plan more practical and appealing.

Our initial scoring structure is:

```text
score(plan) =
    hard_constraint_penalties
    + macro_deviation_penalty
    + ingredient_waste_penalty
    + repetition_penalty
    + preparation_time_penalty
```

## Search Algorithms

We plan to implement and compare three search methods from scratch.

### Hill Climbing with Random Restarts

Hill climbing starts with a randomly generated meal plan and accepts changes that improve its score. If the algorithm gets stuck at a plan that cannot be improved with small changes, a random restart allows it to begin searching from a new starting point.

### Simulated Annealing

Simulated annealing may accept a worse plan earlier in the search process. This can help it move away from a plan that seems good at first but prevents stronger improvements later on. As the search continues, the algorithm becomes more selective.

### Genetic Algorithm

The genetic algorithm starts with multiple possible meal plans instead of relying on one starting plan. Stronger plans are selected to help create new plans. The algorithm can combine sections of two plans or randomly change individual meals as it searches for better solutions.

## Dataset

We are currently deciding on a public recipe dataset. Our options include Food.com Recipes, RecipeNLG, and USDA FoodData Central.

The dataset should ideally provide information about:

- Calories and macronutrients
- Ingredients included in each recipe
- Preparation time
- Dietary labels or restrictions
- Cost information, if available

If cost is not directly available in the selected dataset, we may need to simplify that part of the objective function or estimate costs using ingredient information.

## Evaluation Plan

We will compare the three search algorithms by looking at the quality of the final meal plans they produce. We want to see whether each algorithm creates plans that satisfy the required constraints, reaches a strong objective score, and does so within a reasonable amount of time.

Because local search can produce different results depending on where it starts, we plan to run each algorithm multiple times. This will allow us to compare their performance more fairly rather than relying on one result from each method.

## Planned Repository Structure

```text
cs4100-meal-planner/
├── data/
├── src/
│   ├── search/
│   └── evaluation/
├── notebooks/
├── tests/
├── requirements.txt
└── README.md
```

## Current Progress

- Created the GitHub repository
- Finalized the project proposal
- Defined the general approach for representing weekly meal plans
- Identified possible recipe datasets
- Chosen the local search methods we plan to compare
- Started outlining the objective function

## Next Steps

- Choose the final recipe dataset
- Finalize the objective function
- Set up the project folder structure
- Create a code representation for recipes and weekly meal plans
- Begin implementing hill climbing with random restarts
