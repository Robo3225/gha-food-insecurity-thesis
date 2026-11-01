
from pathlib import Path
import pandas as pd
lst = pd.read_csv("data/processed/kenya_lst_panel.csv")
lst["prefix"] = lst["FNID"].str.extract(r"(KE2016C\d{2})")

# 1. Is the drought heat spatially concentrated in the arid north?
print("2022 LST anomaly by prefix (arid C30/31 vs highland C33/34):")
print(lst[lst.year==2022].groupby("prefix")["lst_anomaly"].mean().sort_values(ascending=False).to_string())

# 2. Do the recovered arid units specifically show 2022 heat?
#    (the 29 units from the D20 recovery)
arid = lst[lst["prefix"].isin(["KE2016C308","KE2016C309"])]
print("\narid C308/C309 units, mean LST anomaly by year:")
print(arid.groupby("year")["lst_anomaly"].mean().to_string())

# 3. Sanity: is the whole window cooler, or just some units?
print("\nnational LST anomaly by year (already have, confirming):")
print(lst.groupby("year")["lst_anomaly"].mean().to_string())