"""
======================================================================

build_spi.py

D18 SPI (Standardized Precipitation Index) for rainfall, replacing the
raw z-score anomaly which is invalid for zero-inflated arid rainfall.

Implements the three D18 requirements exactly:

1. ZERO-ADJUSTED MIXED GAMMA. Gamma is defined for x>0 only. Per unit,
   per calendar month:
     - q  = P(rainfall == 0)  (fraction of zero months in the baseline)
     - fit Gamma(shape, scale) to the NONZERO baseline values only
     - mixed CDF: H(x) = q + (1-q) * G(x)   for x > 0
                  H(0) = q                   for x == 0 (use q/2 midpoint)
     - SPI = Phi^-1(H(x))   (inverse standard normal)

2. FITTING-FAILURE FALLBACK. Units/months with too few nonzero baseline
   observations to fit a stable gamma FALL BACK to an empirical-
   percentile-to-normal transform, and every such cell is FLAGGED and
   logged (D11-style), never silently z-scored.

3. FIXED REFERENCE. Gamma params + q are estimated ONCE from the
   1991-2020 baseline, stored as the climatology artifact, and applied
   to the 2018-2023 panel. Not re-estimated per run.

SPI-3 (three-month accumulation) is the primary feature; SPI-1 optional.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# Minimum nonzero baseline observations to attempt a gamma fit.
# Below this, fall back to empirical percentile (D18 req 2).
MIN_NONZERO_FOR_GAMMA = 10

# SPI is clipped to a sane bound; Phi^-1 near 0/1 goes to +/-inf.
SPI_CLIP = 3.5


# ---------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------

def accumulate(df: pd.DataFrame, months: int, value_col: str,
               unit_col: str = "FNID") -> pd.DataFrame:
    """
    Rolling n-month sum of rainfall per unit (SPI-3 => months=3).
    Requires df sorted by unit, year, month with a contiguous series.
    The first (months-1) rows per unit are NaN (insufficient history).
    """
    df = df.sort_values([unit_col, "year", "month"]).copy()
    df[f"acc{months}"] = (
        df.groupby(unit_col)[value_col]
          .transform(lambda s: s.rolling(months, min_periods=months).sum())
    )
    return df


# ---------------------------------------------------------------------
# Fit the SPI reference from the baseline
# ---------------------------------------------------------------------

def fit_spi_reference(
    baseline: pd.DataFrame,
    acc_col: str,
    unit_col: str = "FNID",
) -> pd.DataFrame:
    """
    Per unit, per calendar month, fit the zero-adjusted mixed-gamma
    reference to the ACCUMULATED baseline rainfall. Returns a parameter
    table: unit, month, q, gamma_shape, gamma_scale, method, n_nonzero.

    method is 'gamma' or 'empirical' (fallback). Empirical cells store
    the sorted nonzero values (as a JSON-ish string) is avoided; instead
    empirical cells are handled at apply time from the stored q plus a
    recorded empirical flag - but to keep the reference self-contained we
    store the nonzero baseline sample for empirical cells in a companion
    long table. For simplicity here, empirical cells store shape/scale as
    NaN and the apply step uses the baseline sample passed alongside.
    """
    rows = []
    for (unit, month), g in baseline.groupby([unit_col, "month"]):
        vals = g[acc_col].dropna().values
        if len(vals) == 0:
            continue
        n = len(vals)
        nonzero = vals[vals > 0]
        n_nonzero = len(nonzero)
        q = float((vals == 0).sum()) / n   # P(zero)

        if n_nonzero >= MIN_NONZERO_FOR_GAMMA:
            # Fit gamma to nonzero values (floc=0: standard 2-param gamma)
            try:
                shape, loc, scale = stats.gamma.fit(nonzero, floc=0)
                if not (np.isfinite(shape) and np.isfinite(scale) and scale > 0):
                    raise ValueError("non-finite gamma params")
                method = "gamma"
            except Exception:
                shape, scale, method = np.nan, np.nan, "empirical"
        else:
            shape, scale, method = np.nan, np.nan, "empirical"

        rows.append({
            unit_col: unit, "month": month, "q": q,
            "gamma_shape": shape, "gamma_scale": scale,
            "method": method, "n_nonzero": n_nonzero, "n_total": n,
        })

    return pd.DataFrame(rows)


def build_empirical_samples(
    baseline: pd.DataFrame, acc_col: str, reference: pd.DataFrame,
    unit_col: str = "FNID",
) -> dict:
    """
    For cells flagged 'empirical', store the sorted nonzero baseline
    accumulation sample so the empirical-percentile transform is
    reproducible at apply time.
    """
    emp = reference[reference["method"] == "empirical"][[unit_col, "month"]]
    keys = set(map(tuple, emp.values))
    samples = {}
    for (unit, month), g in baseline.groupby([unit_col, "month"]):
        if (unit, month) in keys:
            v = np.sort(g[acc_col].dropna().values)
            samples[(unit, month)] = v
    return samples


# ---------------------------------------------------------------------
# Apply SPI to the modelling window
# ---------------------------------------------------------------------

def _spi_gamma(x, q, shape, scale):
    """Mixed-CDF SPI for one value via fitted gamma."""
    if np.isnan(x):
        return np.nan
    if x <= 0:
        H = q / 2.0                      # midpoint for the zero atom
    else:
        G = stats.gamma.cdf(x, a=shape, scale=scale)
        H = q + (1.0 - q) * G
    H = min(max(H, 1e-6), 1 - 1e-6)
    return float(np.clip(stats.norm.ppf(H), -SPI_CLIP, SPI_CLIP))


def _spi_empirical(x, q, sample):
    """Empirical-percentile SPI fallback for un-fittable cells."""
    if np.isnan(x):
        return np.nan
    if x <= 0:
        H = q / 2.0
    else:
        nonzero = sample[sample > 0]
        if len(nonzero) == 0:
            return 0.0
        # empirical CDF position of x among nonzero baseline
        rank = np.searchsorted(np.sort(nonzero), x, side="right") / len(nonzero)
        H = q + (1.0 - q) * rank
    H = min(max(H, 1e-6), 1 - 1e-6)
    return float(np.clip(stats.norm.ppf(H), -SPI_CLIP, SPI_CLIP))


def apply_spi(
    panel: pd.DataFrame,
    acc_col: str,
    reference: pd.DataFrame,
    empirical_samples: dict,
    out_col: str = "spi3",
    unit_col: str = "FNID",
) -> pd.DataFrame:
    """
    Compute SPI for each accumulated panel value against the fixed
    reference. Adds out_col and a boolean out_col+'_empirical' flag.
    """
    ref = reference.set_index([unit_col, "month"])
    df = panel.copy()

    spi_vals = np.full(len(df), np.nan)
    emp_flag = np.zeros(len(df), dtype=bool)

    for i, row in enumerate(df.itertuples(index=False)):
        unit = getattr(row, unit_col)
        month = getattr(row, "month")
        x = getattr(row, acc_col)
        if (unit, month) not in ref.index:
            continue
        r = ref.loc[(unit, month)]
        if r["method"] == "gamma":
            spi_vals[i] = _spi_gamma(x, r["q"], r["gamma_shape"], r["gamma_scale"])
        else:
            sample = empirical_samples.get((unit, month), np.array([]))
            spi_vals[i] = _spi_empirical(x, r["q"], sample)
            emp_flag[i] = True

    df[out_col] = spi_vals
    df[out_col + "_empirical"] = emp_flag
    return df
