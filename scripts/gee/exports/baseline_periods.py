"""
======================================================================

baseline_periods.py

D18 fixed climatological baseline periods. Single source of truth for
all anomaly/VCI climatology extractions, so the periods match the
decision log in exactly one place.

Per D18 (settled; MODIS endpoint corrected 2026-08-05):

Governing principle (D18): the drought must not contaminate the
reference period against which it is measured. All three feature
families apply ONE pre-drought cutoff - 2020, the last full pre-drought
year. The drought years are 2021-2023 (early onset begins 2021 in parts
of the region), so every baseline ends at 2020.

  - CHIRPS rainfall : 1991-2020 (WMO climatological normal; its 2020
    endpoint aligns with the pre-drought boundary).
  - MODIS LST       : 2000-2020 (2000 = start of MODIS record; 2020
    chosen to match the CHIRPS cutoff, one consistent boundary).
  - MODIS NDVI/VCI  : 2000-2020 (same MODIS record).

Baselines differ in length (30 vs 21 years) purely by product
availability - a one-line methodology note, not an inconsistency (D18).
Both endpoints are fixed (not "to present") for reproducibility.

Guard (D18 reversal condition): a reference period must fully EXCLUDE
the 2021-2023 drought years. Any baseline extending into 2021 or later
is blocked.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

# Drought years that must be excluded from any reference period (D18).
# Early onset begins 2021 in parts of the region, so the exclusion band
# is 2021-2023 and every baseline ends at 2020.
DROUGHT_YEARS = (2021, 2022, 2023)
PRE_DROUGHT_CUTOFF = 2020   # last full pre-drought year; every baseline ends here

# start_month defaults to 1 (January). MODIS products begin 2000-02-18,
# so their first FULL month is March 2000; start_month=3 makes the stored
# baseline honest ("2000-03" not "2000-01") rather than relying on empty-
# month nulls. CHIRPS begins 1981, so January is fine.
BASELINE_PERIODS = {
    "chirps": {"start": 1991, "start_month": 1, "end": 2020,
               "label": "CHIRPS_1991_2020_WMO"},
    "lst":    {"start": 2000, "start_month": 3, "end": 2020,
               "label": "MODIS_LST_2000-03_2020"},
    "ndvi":   {"start": 2000, "start_month": 3, "end": 2020,
               "label": "MODIS_NDVI_2000-03_2020"},
}


def get_baseline(feature_key: str) -> dict:
    """Return the D18 baseline period for a feature, guarded."""
    if feature_key not in BASELINE_PERIODS:
        raise KeyError(
            f"No D18 baseline for '{feature_key}'. "
            f"Known: {sorted(BASELINE_PERIODS)}"
        )
    b = dict(BASELINE_PERIODS[feature_key])
    b.setdefault("start_month", 1)

    # D18 guard: reference period must exclude the 2021-2023 drought years.
    # Any endpoint at or beyond 2021 reintroduces the self-normalization
    # the long baseline exists to prevent.
    if b["end"] > PRE_DROUGHT_CUTOFF:
        raise RuntimeError(
            f"{feature_key} baseline ends {b['end']}, past the pre-drought "
            f"cutoff {PRE_DROUGHT_CUTOFF}. D18 requires every baseline to "
            f"exclude the drought years {DROUGHT_YEARS}."
        )
    return b
