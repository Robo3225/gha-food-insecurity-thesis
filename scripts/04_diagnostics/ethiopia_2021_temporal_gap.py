from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

raw = pd.read_csv(
    PROJECT_ROOT /
    "data/raw/world_bank/FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv"
)

eth = raw[
    raw["country"].str.lower() == "ethiopia"
].copy()

eth["ipc_present"] = eth["fews_ipc"].notna()

summary = (
    eth.groupby("year_month")
       .agg(
           rows=("fews_ipc", "size"),
           ipc_present=("ipc_present", "sum"),
           ipc_missing=("ipc_present", lambda s: (~s).sum())
       )
)

print(summary.loc["2020_10":"2022_02"])

oddity = eth[
    (eth["year_month"] == "2021_06") &
    (eth["fews_ipc"].notna())
]

print(f"\nNumber of non-missing IPC records in 2021_06: {len(oddity)}")

if len(oddity):
    print(
        oddity[
            [
                "admin_code",
                "admin_name",
                "fews_ipc",
                "fews_ha",
                "fews_ipc_adjusted",
            ]
        ].to_string(index=False)
    )