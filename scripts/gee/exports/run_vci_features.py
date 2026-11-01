"""
======================================================================

run_vci_features.py

Compute VCI for the 2018-2023 modelling window by applying the frozen
NDVI percentile climatology (run_vci_baseline.py) to the EXISTING
Milestone 7.5 NDVI panel.

NO new GEE export: the modelling-window NDVI already exists at
data/processed/kenya_ndvi_panel.csv. This step is pure local
computation - read NDVI panel, apply VCI climatology, write VCI panel.

VCI = (NDVI - p5) / (p95 - p5), clipped to [0,1], per unit per calendar
month (D18). Interpolated NDVI cells (the 13 flagged in M7.5) carry
their flag through so VCI derived from an interpolated NDVI is
auditable.

Process-only (no GEE): python run_vci_features.py

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import argparse

from scripts.gee.exports.build_climatology import apply_vci
from config.countries import get_country


# Resolve --country up front (unknown fails loud).
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--country", required=True)
_country_arg, _ = _ap.parse_known_args()
COUNTRY = get_country(_country_arg.country)["name"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED = PROJECT_ROOT / "data" / "processed"
CLIM_DIR = PROJECT_ROOT / "data" / "climatology"

NDVI_PANEL = PROCESSED / f"{COUNTRY.lower()}_ndvi_panel.csv"
CLIM_FILE = CLIM_DIR / f"{COUNTRY.lower()}_vci_climatology.csv"
OUT_FILE = PROCESSED / f"{COUNTRY.lower()}_vci_panel.csv"


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    section("BUILD VCI FEATURE PANEL (from existing NDVI panel)")

    for f in (NDVI_PANEL, CLIM_FILE):
        if not Path(f).exists():
            raise RuntimeError(f"Required input missing: {f}")

    ndvi = pd.read_csv(NDVI_PANEL)
    print(f"NDVI panel rows : {len(ndvi):,}")
    if "ndvi" not in ndvi.columns:
        raise RuntimeError(f"NDVI panel has no 'ndvi' column. "
                           f"Columns: {list(ndvi.columns)}")

    clim = pd.read_csv(CLIM_FILE)

    section("APPLY VCI (vs 2000-2020 percentile climatology)")
    panel = apply_vci(
        ndvi, clim, value_col="ndvi", out_col="vci", unit_col="FNID",
    )

    v = panel["vci"]
    print(f"VCI range       : {v.min():.3f} .. {v.max():.3f} (should be [0,1])")
    print(f"VCI mean / sd   : {v.mean():.3f} / {v.std():.3f}")

    # drought years should show LOW VCI (stressed vegetation)
    for yr in [2020, 2021, 2022, 2023]:
        m = panel[panel["year"] == yr]["vci"].mean()
        print(f"  {yr} mean VCI : {m:.3f} (drought years expected low)")

    # carry the NDVI interpolation flag through if present
    keep = ["FNID", "year", "month", "vci"]
    if "ndvi_interpolated" in panel.columns:
        keep.append("ndvi_interpolated")
        n_interp = int(panel["ndvi_interpolated"].sum())
        print(f"\nVCI derived from interpolated NDVI: {n_interp} cells "
              f"(flag carried through)")

    panel[keep].to_csv(OUT_FILE, index=False)
    section("DONE")
    print(f"VCI feature panel : {OUT_FILE}")
    print(f"Rows              : {len(panel):,}")


if __name__ == "__main__":
    main()
