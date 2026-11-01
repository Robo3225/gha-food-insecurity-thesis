"""
verify_record_numbers.py

Re-derive the per-country numbers in the technical record directly from
the frozen v2 panels + climatology artifacts, so the record can be
checked against ground truth (not the scratch log). Read-only.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data" / "processed" / "v2"
PROC = ROOT / "data" / "processed"

def line(): print("-"*66)

for c in ["kenya","ethiopia","somalia"]:
    print("="*66); print(c.upper()); print("="*66)
    panel = pd.read_csv(V2 / f"{c}_feature_panel.csv")
    print(f"v2 rows: {len(panel):,}  cols: {len(panel.columns)}  "
          f"units: {panel['canonical_admin_code'].nunique()}")

    # SPI empirical rows
    if "spi3_empirical" in panel.columns:
        print(f"spi3_empirical rows: {int(panel['spi3_empirical'].sum())}")

    # LST cloudy nulls (lst_anomaly NaN where not a warmup lag issue - use raw)
    if "lst_anomaly" in panel.columns:
        print(f"lst_anomaly nulls (cloudy value-nulls): {int(panel['lst_anomaly'].isna().sum())}")

    # NDVI interpolated / unfillable
    if "ndvi_interpolated" in panel.columns:
        print(f"ndvi_interpolated: {int(panel['ndvi_interpolated'].sum())}")
    if "ndvi_unfillable" in panel.columns:
        print(f"ndvi_unfillable: {int(panel['ndvi_unfillable'].sum())}")
    else:
        print("ndvi_unfillable: (column absent - Kenya frozen pre-flag)")

    # pop_density range + median
    if "pop_density" in panel.columns:
        pd_ = panel.drop_duplicates("canonical_admin_code")["pop_density"]
        print(f"pop_density: {pd_.min():.2f} .. {pd_.max():.1f}  median {pd_.median():.1f}")

    # Koppen modal set
    if "climate_class" in panel.columns:
        cc = sorted(panel["climate_class"].dropna().astype(int).unique())
        print(f"climate_class modal set: {cc} ({len(cc)} classes)")

    # SPI drought signal: 2022 mean + units<=-2
    if "spi3" in panel.columns and "year_month" in panel.columns:
        panel["yr"] = panel["year_month"].str[:4].astype(int)
        m2022 = panel[panel.yr==2022]["spi3"].mean()
        n2 = int((panel[panel.yr==2022]["spi3"]<=-2).sum())
        print(f"2022 SPI mean: {m2022:+.2f}   units<=-2 in 2022: {n2}")

    # VCI 2022 trough
    if "vci" in panel.columns:
        v2020 = panel[panel.yr==2020]["vci"].mean()
        v2022 = panel[panel.yr==2022]["vci"].mean()
        print(f"VCI 2020: {v2020:.3f}  2022: {v2022:.3f}")
    line()
