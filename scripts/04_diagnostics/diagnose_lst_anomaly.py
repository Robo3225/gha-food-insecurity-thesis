

from pathlib import Path
import pandas as pd
clim = pd.read_csv("data/climatology/kenya_lst_climatology.csv")

empty = clim[clim["n_years"] == 0]
print("fully-empty cells:", len(empty))

# Which months? (cloud is seasonal)
print("\nby calendar month:")
print(empty["month"].value_counts().sort_index())

# Which units? (cloud is spatial - highland/coastal)
empty["prefix"] = empty["FNID"].str.extract(r"(KE2016C\d{2})")
print("\nby region prefix:")
print(empty["prefix"].value_counts())

# Are whole units empty, or specific months of specific units?
per_unit = empty.groupby("FNID").size()
print(f"\nunits with any empty cell: {per_unit.count()}")
print(f"units fully empty (all 12 months): {(per_unit == 12).sum()}")
print("distribution of empty-months-per-affected-unit:")
print(per_unit.value_counts().sort_index())

# And the thin (1-19 yr) cells - same pattern?
thin = clim[(clim["n_years"] > 0) & (clim["n_years"] < 20)]
print(f"\nthin cells (1-19 yr): {len(thin)}, months:")
print(thin["month"].value_counts().sort_index())