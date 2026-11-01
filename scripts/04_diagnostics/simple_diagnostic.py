import pandas as pd
# The panel was interpolated but not saved (crash before write). Re-inspect
# via the masked-cells report + check neighbours in the raw composites.
mc = pd.read_csv("data/processed/ethiopia_ndvi_masked_cells.csv")
print("masked-cell report for the 2 unfilled:")
print(mc[mc.FNID.isin(["ET2021C1071012","ET2021C1160035"])].to_string())
print("\nfull masked-cell report shape:", mc.shape)
print("\nany multi-month clusters? masked cells per unit:")
print(mc.groupby("FNID").size().sort_values(ascending=False).head())