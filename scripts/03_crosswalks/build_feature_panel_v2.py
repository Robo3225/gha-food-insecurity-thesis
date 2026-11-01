"""
======================================================================

build_feature_panel_v2.py

Assemble the full Kenya M8 feature panel and freeze it as v2.

Joins, onto the label panel (via the FNID<->admin_code crosswalk):
  DYNAMIC (per unit-month, + their D19 uniform lags 1,3,6,9,12):
    - NDVI                       (M7.5)
    - rainfall + spi3            (D18 SPI)
    - lst_celsius + lst_anomaly  (D20 relaxed mask)
    - vci                        (D18 percentile)
  STATIC (per unit, time-invariant):
    - pop_density                (D21 WorldPop 2020)
    - climate_class              (D21 amendment: Koppen-Geiger)

Join discipline (M8 brief):
  - route feature FNID -> admin_code via the crosswalk, join onto the
    label panel's canonical_admin_code (as build_model_panel does)
  - how="left" from the label panel (639 units), so FSC-only polygons
    (Mvita) drop; every label row must find each feature or it's a hard
    failure (fail loud on unmatched, per the brief)
  - assert row count after EACH stream, so a drop is caught at the
    stream that caused it, not at the end
  - lag warmup NaNs (first 12 months per unit) and QC-null cells are
    EXPECTED nulls, distinguished from JOIN-failure nulls: the join
    match is checked on the raw feature column, not the lags

Freeze:
  - data/processed/v2/kenya_feature_panel.csv
  - data/processed/v2/MANIFEST.txt  (row/unit/month counts, feature
    list, source panel hashes, git commit, build timestamp)

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import argparse

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# expected_units comes from the enhanced country config (labelled_units),
# the same single source the --country resolver uses - not crosswalk_config,
# so all three countries' counts are guaranteed consistent.
#
# This script runs as a plain script from 03_crosswalks (not -m), so the
# project root is NOT on sys.path by default - insert it before the
# absolute config import (same guard as build_crosswalk.py).
import sys as _sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from config.countries import get_country as _get_country

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--country", required=True)
_country_arg, _ = _ap.parse_known_args()
COUNTRY = _get_country(_country_arg.country)["name"]
expected_units = _get_country(_country_arg.country)["labelled_units"]
PANEL_START, PANEL_END = "2018-01", "2023-12"
N_MONTHS = 72
LAGS = (1, 3, 6, 9, 12)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
TEMPORAL = PROCESSED / "temporal"
CROSSWALK = (PROJECT_ROOT / "data" / "crosswalks"
             / f"{COUNTRY.lower()}_fnid_admin_code_crosswalk.csv")
LABEL_FILE = PROCESSED / f"{COUNTRY.lower()}_label_panel.csv"

V2 = PROCESSED / "v2"
V2.mkdir(parents=True, exist_ok=True)
OUT_FILE = V2 / f"{COUNTRY.lower()}_feature_panel.csv"
MANIFEST = V2 / "MANIFEST.txt"


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# =====================================================================
# Dynamic-feature streams: (temporal-lagged file, raw value col, extra cols)
# =====================================================================

DYNAMIC = {
    "ndvi":  (PROCESSED / f"{COUNTRY.lower()}_ndvi_panel.csv", "ndvi",
              ["ndvi_interpolated", "ndvi_unfillable"], False),  # NDVI: level only, no lags (D19)
    "spi3":  (TEMPORAL / f"{COUNTRY.lower()}_spi3_lagged.csv", "spi3",
              ["rainfall", "spi3_empirical"], True),
    "lst":   (TEMPORAL / f"{COUNTRY.lower()}_lst_anomaly_lagged.csv", "lst_anomaly",
              ["lst_celsius"], True),
    # VCI derives from NDVI; it should NOT re-carry ndvi_interpolated
    # (that flag already arrives with the NDVI stream). Carrying it here
    # produces ndvi_interpolated_x/_y suffix collision. Drop it from the
    # VCI extra cols; if a VCI-specific interpolation flag exists, name it
    # 'vci_interpolated' in the VCI panel instead.
    "vci":   (TEMPORAL / f"{COUNTRY.lower()}_vci_lagged.csv", "vci",
              [], True),
}
# NOTE: NDVI itself is the vegetation LEVEL; D19 amendment lags the three
# ANOMALY/index features (spi3, lst_anomaly, vci). NDVI enters as level
# only unless the temporal step also lagged it. Adjust `has_lags` per your
# temporal build.

STATIC_FILE = PROCESSED / f"{COUNTRY.lower()}_static_panel.csv"
STATIC_COLS = ["pop_density", "climate_class"]


def join_stream(panel, crosswalk, name, path, raw_col, extra_cols, has_lags):
    if not path.exists():
        raise RuntimeError(f"[{name}] feature panel missing: {path}")
    feat = pd.read_csv(path)

    feat = feat.merge(crosswalk, on="FNID", how="inner")
    feat["year_month"] = (
        feat["year"].astype(int).astype(str) + "_"
        + feat["month"].astype(int).astype(str).str.zfill(2)
    )

    cols = [raw_col] + list(extra_cols)
    if has_lags:
        cols += [f"{raw_col}_lag{k}" for k in LAGS]
    cols = [c for c in cols if c in feat.columns]

    before = len(panel)
    panel = panel.merge(
        feat[["admin_code", "year_month", *cols]],
        left_on=["canonical_admin_code", "year_month"],
        right_on=["admin_code", "year_month"],
        how="left", validate="one_to_one", indicator="_merge_ind",
    ).drop(columns=["admin_code"])

    if len(panel) != before:
        raise RuntimeError(
            f"[{name}] join changed row count {before}->{len(panel)}."
        )

    # Distinguish JOIN FAILURE from a legitimately-null VALUE.
    #   - join failure: the label row found no matching feature row at all
    #     (crosswalk/hierarchy mismatch) -> _merge_ind == "left_only".
    #   - value null: the row joined fine but the feature VALUE was already
    #     NaN (e.g. LST QC-masked cloudy cells, or a lag reaching a cloudy
    #     month). These are EXPECTED and must not fail the assertion.
    # Checking raw_col.isna() conflates the two; the merge indicator
    # separates them.
    n_join_fail = (panel["_merge_ind"] == "left_only").sum()
    n_value_null = panel[raw_col].isna().sum()
    panel = panel.drop(columns=["_merge_ind"])

    print(f"  [{name}] joined {len(cols)} cols; join-failures: {n_join_fail}"
          f"; value-nulls (expected, e.g. cloud): {n_value_null}")

    if n_join_fail:
        raise RuntimeError(
            f"[{name}] {n_join_fail} label rows found NO matching feature "
            f"row - crosswalk/hierarchy mismatch. (Distinct from the "
            f"{n_value_null} legitimately-null values, which are fine.)"
        )
    return panel


def main():
    section(f"ASSEMBLE {COUNTRY.upper()} FEATURE PANEL v2")

    crosswalk = pd.read_csv(CROSSWALK)[["admin_code", "FNID"]]
    labels = pd.read_csv(LABEL_FILE)
    n_units = labels["canonical_admin_code"].nunique()
    if n_units != expected_units:
        raise RuntimeError(f"Label panel {n_units} units, expected {expected_units}.")
    print(f"Label panel: {len(labels):,} rows, {n_units} units")

    panel = labels.copy()

    section("JOIN DYNAMIC FEATURES (+ lags)")
    for name, (path, raw, extra, has_lags) in DYNAMIC.items():
        panel = join_stream(panel, crosswalk, name, path, raw, extra, has_lags)

    section("JOIN STATIC FEATURES")
    if STATIC_FILE.exists():
        static = pd.read_csv(STATIC_FILE).merge(crosswalk, on="FNID", how="inner")
        scols = [c for c in STATIC_COLS if c in static.columns]
        before = len(panel)
        panel = panel.merge(
            static[["admin_code", *scols]],
            left_on="canonical_admin_code", right_on="admin_code",
            how="left", validate="many_to_one",
        ).drop(columns=["admin_code"])
        if len(panel) != before:
            raise RuntimeError(f"static join changed rows {before}->{len(panel)}")
        for c in scols:
            n_missing = panel[c].isna().sum()
            print(f"  [static:{c}] unmatched: {n_missing}")
            if n_missing:
                raise RuntimeError(f"[static:{c}] {n_missing} units unmatched.")
    else:
        print(f"  \u26a0 static panel not found ({STATIC_FILE}); "
              f"freezing WITHOUT static features (add via amendment when ready).")

    section("VALIDATE")
    expected_rows = expected_units * N_MONTHS
    print(f"Rows: {len(panel):,} (expect {expected_rows:,})")
    if len(panel) != expected_rows:
        raise RuntimeError(f"panel {len(panel)} rows, expected {expected_rows}")
    print(f"Units: {panel['canonical_admin_code'].nunique()}  Columns: {len(panel.columns)}")

    section("FREEZE v2")
    panel.to_csv(OUT_FILE, index=False)

    source_files = [LABEL_FILE, CROSSWALK] + [p for (p, *_ ) in DYNAMIC.values()]
    if STATIC_FILE.exists():
        source_files.append(STATIC_FILE)

    with open(MANIFEST, "w") as f:
        f.write(f"{COUNTRY} feature panel v2\n")
        f.write(f"built: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"git commit: {git_commit()}\n")
        f.write(f"rows: {len(panel):,}  units: {expected_units}  months: {N_MONTHS}\n")
        f.write(f"window: {PANEL_START}..{PANEL_END}\n")
        f.write(f"lags: {LAGS}\n\n")
        f.write(f"columns ({len(panel.columns)}):\n")
        for c in panel.columns:
            f.write(f"  {c}\n")
        f.write(f"\nsource file hashes (sha256[:16]):\n")
        for p in source_files:
            f.write(f"  {sha256(Path(p))}  {Path(p).name}\n")
        f.write(f"\noutput hash: {sha256(OUT_FILE)}  {OUT_FILE.name}\n")

    print(f"Panel    : {OUT_FILE}")
    print(f"Manifest : {MANIFEST}")
    print(f"Output hash: {sha256(OUT_FILE)}")
    print("\n\u2713 v2 frozen.")


if __name__ == "__main__":
    main()
