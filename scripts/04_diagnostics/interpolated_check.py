import pandas as pd

p = pd.read_csv("data/processed/kenya_model_panel.csv")
print("interpolated in panel:", p["ndvi_interpolated"].sum())
# cross-check against the masked-cells report
m = pd.read_csv("data/processed/kenya_ndvi_masked_cells.csv")
print("masked cells reported:", len(m), "interpolated:", m["interpolated"].sum())