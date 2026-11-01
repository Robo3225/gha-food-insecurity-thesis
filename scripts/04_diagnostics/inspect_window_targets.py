# Inspect window targets

from pathlib import Path

import pandas as pd


p = pd.read_csv("data/processed/kenya_model_panel.csv")
p["year"] = p["year_month"].str[:4].astype(int)
# observed-only positives per year — strips forward-fill
obs = p[p["is_observed"]]
for yr in [2021, 2022, 2023]:
    w = obs[obs["year"] == yr]
    print(yr, "observed positives:", int(pd.to_numeric(w["phase3plus"], errors="coerce").sum()),
          "of", w["phase3plus"].notna().sum())