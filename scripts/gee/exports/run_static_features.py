"""
======================================================================

run_static_features.py

Extract the D21 static features for a country and write the static
panel: one row per unit with population density and dominant Köppen
climate class.

  pop_density   (D21)           : WorldPop 2020, total/area (people/km2)
  climate_class (D21 amendment) : Köppen-Geiger present-day dominant class

Single-shot (no monthly loop, no baseline): two zonal reductions over
fixed rasters, merged on FNID. Output feeds build_feature_panel_v2.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

import pandas as pd

import ee

from config.config import PROJECT_ID
from config.countries import get_country

from scripts.gee.extractors.gee_utils import initialize, load_feature_collection
from scripts.gee.exports.build_static import (
    build_population_density,
    build_dominant_climate,
    collect_climate_classes,
    assert_shared_taxonomy,
    KOPPEN_ASSET,
    KOPPEN_BAND,
    KOPPEN_VALID_CODES,
)


# Resolve --country up front (unknown fails loud).
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--country", required=True)
_country_arg, _ = _ap.parse_known_args()
_CFG = get_country(_country_arg.country)
COUNTRY = _CFG["name"]

# WorldPop 2020 resolved per country (ImageCollection filtered -> .first()).
WORLDPOP_COLLECTION = "WorldPop/GP/100m/pop"
WORLDPOP_COUNTRY = _CFG["iso3"]   # KEN/ETH/SOM from config
WORLDPOP_YEAR = 2020

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
OUT_FILE = PROCESSED / f"{COUNTRY.lower()}_static_panel.csv"


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def fc_to_df(fc, columns):
    """Pull a FeatureCollection's properties to a DataFrame via getInfo."""
    feats = fc.getInfo()["features"]
    rows = [{c: f["properties"].get(c) for c in columns} for f in feats]
    return pd.DataFrame(rows)


def main():
    section(f"EXTRACT {COUNTRY.upper()} STATIC FEATURES")

    initialize(PROJECT_ID)
    polygons = load_feature_collection(_CFG["asset"])

    # --- population density ------------------------------------------
    section("POPULATION DENSITY (WorldPop 2020)")
    wp = (ee.ImageCollection(WORLDPOP_COLLECTION)
          .filter(ee.Filter.eq("country", WORLDPOP_COUNTRY))
          .filter(ee.Filter.eq("year", WORLDPOP_YEAR))
          .first())
    if wp is None:
        raise RuntimeError(
            f"WorldPop image not found for {WORLDPOP_COUNTRY} {WORLDPOP_YEAR}."
        )
    pop_fc = build_population_density(polygons, wp, id_property="FNID")
    pop_df = fc_to_df(pop_fc, ["FNID", "pop_total", "area_km2", "pop_density"])
    print(f"units: {len(pop_df):,}")
    print(f"pop_density range: {pop_df['pop_density'].min():.2f} .. "
          f"{pop_df['pop_density'].max():.1f} people/km2")
    # sanity: density ordering (sparse pastoral low, urban high)
    print(f"median density: {pop_df['pop_density'].median():.1f}")

    # --- dominant Köppen climate class -------------------------------
    section("DOMINANT KÖPPEN CLIMATE CLASS")
    clim_fc = build_dominant_climate(
        polygons, KOPPEN_ASSET, id_property="FNID", band=KOPPEN_BAND
    )
    clim_df = fc_to_df(clim_fc, ["FNID", "climate_class"])
    # Two-step cast: -> float (tolerates None from units with no modal
    # class) -> nullable Int64 (tolerates the resulting NaN). A direct
    # object->Int64 cast fails when getInfo() returns None for a unit
    # whose small polygon caught no 1km Köppen pixel centroid.
    clim_df["climate_class"] = (
        pd.to_numeric(clim_df["climate_class"], errors="coerce")
          .astype("float")
          .round()
          .astype("Int64")
    )
    n_null_class = clim_df["climate_class"].isna().sum()
    print(f"units: {len(clim_df):,}")
    if n_null_class:
        print(f"\u26a0 {n_null_class} units returned NO modal Köppen class "
              f"(small polygon vs 1km grid). Listed below; these need a "
              f"nearest-pixel fallback or a documented exclusion before freeze:")
        print(clim_df[clim_df["climate_class"].isna()]["FNID"].to_string(index=False))

    modal = collect_climate_classes(clim_df)
    print(f"dominant classes present (unit-level modal set): {sorted(modal)}")
    # validate codes are in-range (single-country form of the D21 check)
    invalid = modal - KOPPEN_VALID_CODES
    if invalid:
        raise RuntimeError(
            f"climate classes {sorted(invalid)} outside Köppen 1-30 - "
            f"wrong band/raster or nodata sentinel."
        )
    n_missing = clim_df["climate_class"].isna().sum()
    if n_missing:
        print(f"\u26a0 {n_missing} units have no modal class (tiny polygon vs "
              f"1km grid?) - investigate before freeze.")

    # --- merge --------------------------------------------------------
    section("MERGE STATIC PANEL")
    static = pop_df.merge(clim_df, on="FNID", how="outer", validate="one_to_one")
    print(f"static panel rows: {len(static):,} (units)")

    miss_pop = static["pop_density"].isna().sum()
    miss_clim = static["climate_class"].isna().sum()
    print(f"missing pop_density: {miss_pop}   missing climate_class: {miss_clim}")

    static = static[["FNID", "pop_density", "pop_total", "area_km2", "climate_class"]]
    static.to_csv(OUT_FILE, index=False)

    section("DONE")
    print(f"Static panel : {OUT_FILE}")
    print(f"Rows (units) : {len(static):,}")
    print("\nNote: the cross-country taxonomy check (assert_shared_taxonomy)")
    print("runs when all three countries' static panels exist - it compares")
    print("the per-country modal class sets to confirm shared feature levels.")


if __name__ == "__main__":
    main()
