# Data

The recipe data is **not committed** to this repo (the files are ~800MB total
and exceed GitHub's limits), so each person downloads their own copy here.

## Source

[Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) (Kaggle)

## Setup

1. Create a Kaggle API token: Kaggle → Account → "Create New API Token".
   This downloads `kaggle.json`. Move it to `~/.kaggle/kaggle.json` and
   restrict it: `chmod 600 ~/.kaggle/kaggle.json`.

2. From the repo root, with the venv active:

kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions -p data/ --unzip

3. Verify the loader reads it:

python src/recipe.py data/RAW_recipes.csv

Expected: `Loaded 231637 recipes` followed by one printed Recipe.

## Files

The loader uses **`RAW_recipes.csv`** (human-readable: name, minutes,
nutrition, ingredients, tags). The download includes several other files
we don't use:

- `PP_recipes.csv` / `PP_users.csv` — preprocessed/tokenized, integer IDs
  only, no readable nutrition. Not used.
- `RAW_interactions.csv`, `interactions_*.csv` — user ratings/reviews.
  Not used yet; a possible future soft-goal signal (favor higher-rated recipes).