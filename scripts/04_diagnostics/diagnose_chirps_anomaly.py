"""
diagnose_chirps_anomaly.py

Investigates implausible CHIRPS rainfall anomalies.

Checks:
1. Climatology standard deviation distribution.
2. Scale consistency between climatology and modelling panel.
3. Duplicate / missing climatology keys.
4. Extreme anomaly rows.
5. Whether extremes are driven by tiny climatological variance.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CLIM = ROOT / "data" / "climatology" / "kenya_chirps_climatology.csv"
PANEL = ROOT / "data" / "processed" / "kenya_chirps_panel.csv"

clim = pd.read_csv(CLIM)
panel = pd.read_csv(PANEL)

print("=" * 70)
print("CHIRPS ANOMALY DIAGNOSTIC")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Climatology SD diagnostics
# ------------------------------------------------------------------

print("\n1. CLIMATOLOGY STANDARD DEVIATION")

print(clim["clim_sd"].describe().round(3))

for t in [0, 0.01, 0.1, 1, 5]:
    print(f"cells with clim_sd < {t:<4}: {(clim['clim_sd'] < t).sum():,}")

print(f"cells with clim_sd == 0 : {(clim['clim_sd'] == 0).sum():,}")

# ------------------------------------------------------------------
# 2. Rainfall scale check
# ------------------------------------------------------------------

print("\n2. SCALE CHECK")

print(
    f"Baseline clim_mean : "
    f"{clim.clim_mean.min():.2f} .. {clim.clim_mean.max():.2f}"
)

print(
    f"Panel rainfall     : "
    f"{panel.rainfall.min():.2f} .. {panel.rainfall.max():.2f}"
)

# ------------------------------------------------------------------
# 3. Climatology integrity
# ------------------------------------------------------------------

print("\n3. CLIMATOLOGY INTEGRITY")

dup = clim.duplicated(["FNID", "month"]).sum()

print(f"Duplicate FNID-month keys : {dup:,}")

merged = panel.merge(
    clim,
    on=["FNID", "month"],
    how="left",
    indicator=True,
)

missing = (merged["_merge"] != "both").sum()

print(f"Missing climatology rows  : {missing:,}")

# ------------------------------------------------------------------
# 4. Extreme anomalies
# ------------------------------------------------------------------

print("\n4. TOP 10 EXTREME ANOMALIES")

extreme = (
    merged.reindex(
        merged["rainfall_anomaly"]
        .abs()
        .sort_values(ascending=False)
        .index
    )
    .head(10)
)

cols = [
    "FNID",
    "year",
    "month",
    "rainfall",
    "clim_mean",
    "clim_sd",
    "rainfall_anomaly",
]

print(extreme[cols].to_string(index=False))

# ------------------------------------------------------------------
# 5. Large anomaly summary
# ------------------------------------------------------------------

print("\n5. LARGE ANOMALY SUMMARY")

for t in [3, 5, 10, 20, 100]:
    n = (merged["rainfall_anomaly"].abs() > t).sum()
    print(f"|anomaly| > {t:<3}: {n:,}")

print("\nAffected months:")

print(
    merged.loc[
        merged["rainfall_anomaly"].abs() > 10,
        "month",
    ]
    .value_counts()
    .sort_index()
)

print("\nAffected FNIDs:")

print(
    merged.loc[
        merged["rainfall_anomaly"].abs() > 10,
        "FNID",
    ]
    .value_counts()
    .head(20)
)

print("\n" + "=" * 70)
print("Diagnostic complete.")
print("=" * 70)