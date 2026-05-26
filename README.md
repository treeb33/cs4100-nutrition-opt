# Adaptive Meal Planner: Local Search for Weekly Meal Planning

## CS4100 Final Project

**Team Members:** Sanah Menon, Triya Basu, Isabella Pozzi

## Project Overview

Meal planning can be difficult because people have to balance nutrition goals, dietary restrictions, grocery budget, cooking time, and ingredient waste at the same time. A plan may look good nutritionally, but still be unrealistic if it is too expensive, takes too long to prepare, or requires buying many ingredients that are barely reused.

For this project, we are building a system that generates a weekly meal plan using local search. Instead of choosing each meal individually, the system evaluates the full week together and searches for a plan that is both valid and realistic for the user.

## Problem Representation

A single state is one complete weekly meal plan made up of 21 meal slots: breakfast, lunch, and dinner across 7 days. Each slot is filled with a recipe from a public recipe dataset.

A neighboring state will be created by making a small change to the current plan, such as replacing one meal with another recipe. This allows the search algorithms to gradually improve a plan rather than trying every possible combination of meals.

## Constraints and Objective Function

The system will score each weekly plan based on both hard constraints and softer preferences.

**Hard constraints may include:**
- Daily calorie range
- Allergen or dietary restrictions
- Weekly grocery budget

**Soft goals may include:**
- Staying close to macro targets
- Reusing ingredients to reduce waste
- Avoiding too much repetition
- Keeping preparation time realistic

A lower score will represent a better meal plan. Hard constraint violations will receive much larger penalties than soft goal issues so that the system prioritizes producing valid plans.

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

We plan to implement and compare three search methods from scratch:

### Hill Climbing with Random Restarts
Starts with a random meal plan and repeatedly accepts changes that improve the score. Random restarts help the algorithm try new starting points if it gets stuck.

### Simulated Annealing
Sometimes accepts a worse plan early in the search, which may help it move out of local optima and eventually find a stronger solution.

### Genetic Algorithm
Starts with multiple meal plans and improves them through selection, crossover, and mutation. This allows the algorithm to explore different possible plans at the same time.

## Dataset

We are currently deciding on a public recipe dataset. Our options include Food.com Recipes, RecipeNLG, and USDA FoodData Central.

The dataset should ideally provide:
- Calories and macronutrients
- Ingredients
- Preparation time
- Dietary information
- Cost information, if available

## Evaluation Plan

We will compare the three search algorithms based on:

- Final objective score
- Constraint satisfaction rate
- Runtime or convergence speed
- Overall practicality of the generated plans

Because local search results can depend on the initial starting point, we plan to run each algorithm multiple times and compare its overall performance.

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
- Finalized the project proposal and general approach
- Identified possible recipe datasets
- Selected the three local search methods to compare
- Began outlining the objective function

## Next Steps

- Choose the final recipe dataset
- Finalize the objective function
- Set up the project folder structure
- Represent recipes and weekly meal plans in code
- Begin implementing hill climbing with random restarts
