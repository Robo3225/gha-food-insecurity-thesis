import pandas as pd
lst = pd.read_csv("data/processed/ethiopia_lst_panel.csv")
cw = pd.read_csv("data/crosswalks/ethiopia_fnid_admin_code_crosswalk.csv")[["FNID","admin_code"]]
ru = pd.read_csv("data/interim/ethiopia_fews_reporting_units.csv")[["admin_code","ADMIN1"]].rename(columns={"ADMIN1":"region"})

m = lst.merge(cw,on="FNID").merge(ru,on="admin_code")

# 2022 LST anomaly by region — arid east should be WARM (positive) if the
# heat signal mirrors the rainfall drought; highlands cool/negative
print("2022 mean LST anomaly by region (warmest first):")
print(m[m.year==2022].groupby("region")["lst_anomaly"].mean().sort_values(ascending=False).round(3).to_string())

# cross-check: are the arid-east regions (that were driest in SPI) the warmest in LST?
print("\narid-east regions (Somali/Afar/Harari/Dire Dawa) LST anomaly by year:")
arid = m[m.region.isin(["somali","afar","harari","dire dawa"])]
print(arid.groupby("year")["lst_anomaly"].mean().round(3).to_string())