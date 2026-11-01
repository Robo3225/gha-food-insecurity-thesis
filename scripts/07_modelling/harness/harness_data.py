"""
======================================================================

harness_data.py  (M10.1 — data loading + D22 identity enforcement)

Loads the three frozen v2 feature panels and enforces feature-vector
identity at load time via the D22 canonical predictor list, with a
fail-loud assertion. No consolidated "v3" file exists or is needed —
identity is enforced here, not by merging panels (D22).

NO MODELLING. This module only loads, verifies, and shapes data.

Canonical predictor list (D22) — model inputs, and ONLY these:
  ndvi
  spi3 + lags {1,3,6,9,12}
  lst_anomaly + lags {1,3,6,9,12}
  vci + lags {1,3,6,9,12}
  pop_density  (log-transformed here — right-skewed, D22)
  climate_class  (categorical)

Everything else in the panels (LZNAME, admin codes, year_month, raw
rainfall/lst_celsius, all audit flags) is NON-predictor and excluded.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / "data" / "processed" / "v2"

COUNTRIES = ["kenya", "ethiopia", "somalia"]

LAGS = (1, 3, 6, 9, 12)

# ---- D22 canonical predictor list ----------------------------------
LAGGED = ["spi3", "lst_anomaly", "vci"]
CANONICAL_PREDICTORS = (
    ["ndvi"]
    + [f"{f}_lag{k}" for f in LAGGED for k in LAGS]
    + LAGGED                       # contemporaneous value of each lagged feature
    + ["pop_density", "climate_class"]
)
# note: contemporaneous spi3/lst_anomaly/vci ARE predictors (level +
# lags). ndvi is level-only. Ordering fixed for reproducibility.

# columns needed by the harness that are NOT predictors (identity keys,
# target, baselines, temporal index)
TARGET_BINARY = "phase3plus"
TARGET_ORDINAL = "fews_ipc"          # ordinal phase for TFT (D9)
BENCHMARK_PROJ = "fews_proj_near"    # FEWS NET projection baseline (D3)
IS_OBSERVED = "is_observed"
UNIT_COL = "canonical_admin_code"
TIME_COL = "year_month"

# pop_density is log-transformed at modelling (D22); log1p (handles 0)
LOG_TRANSFORM = ["pop_density"]


def _panel_path(slug):
    return V2 / f"{slug}_feature_panel.csv"


def load_country(slug: str) -> pd.DataFrame:
    p = _panel_path(slug)
    if not p.exists():
        raise RuntimeError(f"Frozen panel missing: {p}")
    df = pd.read_csv(p)
    df["country"] = slug
    return df


def assert_canonical_predictors(panels: dict) -> None:
    """
    D22 fail-loud identity enforcement. Every country panel must contain
    EXACTLY the canonical predictor set (no missing, and we do not care
    about extra metadata columns — those are excluded downstream, but a
    MISSING predictor is a hard failure).
    """
    required = set(CANONICAL_PREDICTORS)
    for slug, df in panels.items():
        have = set(df.columns)
        missing = required - have
        if missing:
            raise RuntimeError(
                f"[{slug}] MISSING canonical predictors {sorted(missing)}. "
                f"Feature-vector identity (D22) violated — cannot proceed."
            )
    # also assert the predictor columns are dtype-compatible across
    # countries (all numeric except climate_class which is categorical int)
    print(f"\u2713 D22: all {len(panels)} panels contain the "
          f"{len(CANONICAL_PREDICTORS)}-column canonical predictor set.")


def _check_required_nonpredictors(panels: dict) -> None:
    """Target, baselines, and temporal keys must be present too."""
    # NB: the FEWS projection (BENCHMARK_PROJ) is deliberately NOT a panel
    # column — per D24 it is joined at the harness (harness_fews) from the
    # raw World Bank source, leaving the frozen panels untouched. So it is
    # not required here; only genuine panel columns are.
    need = [TARGET_BINARY, TARGET_ORDINAL, IS_OBSERVED, UNIT_COL, TIME_COL]
    for slug, df in panels.items():
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise RuntimeError(
                f"[{slug}] missing required non-predictor columns {miss} "
                f"(target/baseline/temporal keys)."
            )


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """log1p the right-skewed density feature (D22). Monotonic, threshold-
    free, applied identically to every country — preserves C3-cleanliness."""
    for col in LOG_TRANSFORM:
        if col in df.columns:
            if (df[col] < 0).any():
                raise RuntimeError(f"{col} has negatives; log1p invalid.")
            df[col] = np.log1p(df[col])
    return df


def add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Integer month index for chronology assertions and walk-forward
    splitting. Parsed from year_month 'YYYY_MM' -> months since 2018-01.
    """
    ym = df[TIME_COL].astype(str)
    year = ym.str[:4].astype(int)
    month = ym.str[5:7].astype(int)
    df["t_index"] = (year - 2018) * 12 + (month - 1)
    df["year"] = year
    df["month"] = month
    return df


def load_all() -> pd.DataFrame:
    """
    Load all three panels, enforce D22 identity, apply the log-transform,
    add the time index, and return a single stacked frame (country-tagged).
    This is the harness's single entry point for data.
    """
    panels = {slug: load_country(slug) for slug in COUNTRIES}
    assert_canonical_predictors(panels)
    _check_required_nonpredictors(panels)

    frames = []
    for slug, df in panels.items():
        df = apply_log_transform(df)
        df = add_time_index(df)
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)

    print(f"\u2713 loaded {len(allp):,} unit-months across "
          f"{allp['country'].nunique()} countries "
          f"({', '.join(COUNTRIES)}).")
    print(f"  per country: "
          + ", ".join(f"{c}={len(allp[allp.country==c]):,}" for c in COUNTRIES))
    return allp


if __name__ == "__main__":
    df = load_all()
    print("\ncanonical predictors:", CANONICAL_PREDICTORS)
    print(f"predictor count: {len(CANONICAL_PREDICTORS)}")
