"""
======================================================================

build_temporal_features.py

D19 (+ amendment) temporal features: apply the UNIFORM lag set
(1, 3, 6, 9, 12 months) to all three dynamic biophysical features -
SPI-3 rainfall, VCI, and LST anomaly - via the leakage-safe lag
generator.

Per the D19 amendment (2026-08-07): all three dynamic features receive
the same lag set; the operative lag per feature per country is left for
SHAP / TFT variable selection to reveal, not asserted here.

Each feature's lags are generated WITHIN its own panel (contiguous
monthly series per unit), then the lagged columns are what enter the
wide join. The raw contemporaneous value is retained alongside its lags
(short lags carry within-season dynamics; long lags carry the ~1-year
transmission - the amendment keeps all of them).

Leakage safety is enforced by build_lags (within-unit, backward-only,
contiguity-asserted).

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import argparse

from pathlib import Path

import pandas as pd

from scripts.gee.exports.build_lags import add_lags, SPI_LAGS


from config.countries import get_country

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--country", required=True)
_country_arg, _ = _ap.parse_known_args()
COUNTRY = get_country(_country_arg.country)["name"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED = PROJECT_ROOT / "data" / "processed"

# (panel file, value column to lag) for each dynamic feature.
DYNAMIC_FEATURES = {
    "spi3":        (PROCESSED / f"{COUNTRY.lower()}_chirps_panel.csv", "spi3"),
    "lst_anomaly": (PROCESSED / f"{COUNTRY.lower()}_lst_panel.csv", "lst_anomaly"),
    "vci":         (PROCESSED / f"{COUNTRY.lower()}_vci_panel.csv", "vci"),
}

OUT_DIR = PROCESSED / "temporal"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    section(f"BUILD {COUNTRY.upper()} TEMPORAL FEATURES (uniform lags {SPI_LAGS})")

    for name, (path, value_col) in DYNAMIC_FEATURES.items():
        if not path.exists():
            raise RuntimeError(f"Feature panel missing: {path}")

        panel = pd.read_csv(path)
        print(f"\n{name}: {len(panel):,} rows, lagging '{value_col}'")

        lagged = add_lags(
            panel, value_col=value_col, lags=SPI_LAGS,
            unit_col="FNID", time_cols=("year", "month"),
        )

        lag_cols = [f"{value_col}_lag{k}" for k in SPI_LAGS]
        # report how many rows have complete lags (after the 12-mo warmup)
        complete = lagged[lag_cols].notna().all(axis=1).sum()
        print(f"  lag columns: {lag_cols}")
        print(f"  rows with all lags present: {complete:,} "
              f"(first 12 months per unit are NaN by construction)")

        out = OUT_DIR / f"{COUNTRY.lower()}_{name}_lagged.csv"
        lagged.to_csv(out, index=False)
        print(f"  -> {out}")

    section("DONE")
    print("All three dynamic features lagged with the uniform D19 set.")
    print("Operative lag per feature is left for SHAP / variable selection.")


if __name__ == "__main__":
    main()
