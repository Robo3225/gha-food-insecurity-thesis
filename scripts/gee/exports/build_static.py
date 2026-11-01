"""
======================================================================

build_static.py

D21 (+ 2026-08-08 amendment) static features: population density and
dominant Köppen-Geiger climate class, one value per unit (time-
invariant), identical in form across all three countries.

  - Population density (static_real): total unit population / unit area
    (people/km2), from a FIXED WorldPop 2020 raster. Count rejected
    (area-dominated); density is threshold-free and transfers cleanly.

  - Climate zone (static_categorical): dominant Köppen-Geiger present-
    day climate class per unit, mode-reduced. Köppen-Geiger is native
    to GEE with a global shared taxonomy, so categorical levels are
    shared across countries by construction (D21 amendment replaced FAO
    GAEZ with Köppen-Geiger: GAEZ is not native to GEE and its manual
    ingest is a correctness surface not worth the cost for a
    conditioning covariate). The §648 check confirms the class set is
    shared across the three extents rather than assuming it.

WorldPop asset id must still be verified against the live GEE catalog
(the biophysical collections are confirmed; WorldPop is not, from the
dev environment). Köppen-Geiger present-day classification is available
in GEE - confirm the exact asset id below.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import ee


# ---------------------------------------------------------------------
# Population density  (D21, unchanged by the amendment)
# ---------------------------------------------------------------------

def build_population_density(
    polygons: "ee.FeatureCollection",
    worldpop_image_id: str,
    id_property: str = "FNID",
) -> "ee.FeatureCollection":
    """
    Total population per unit (SUM of the WorldPop count raster over the
    polygon) / unit area in km2 -> density (people/km2). WorldPop 'pop'
    rasters are per-pixel COUNTS, so SUM then divide by area (D21).
    """
    pop = ee.Image(worldpop_image_id)

    def reduce_unit(f):
        geom = f.geometry()
        area_km2 = geom.area(maxError=1).divide(1e6)
        total_pop = pop.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=geom,
            scale=100, maxPixels=1e9, bestEffort=True,
        ).values().get(0)
        total_pop = ee.Number(ee.Algorithms.If(total_pop, total_pop, 0))
        return ee.Feature(None, {
            id_property: f.get(id_property),
            "pop_total": total_pop,
            "area_km2": area_km2,
            "pop_density": total_pop.divide(area_km2),
        })

    return polygons.map(reduce_unit)


# ---------------------------------------------------------------------
# Köppen-Geiger climate zone  (D21 amendment: replaces GAEZ)
# ---------------------------------------------------------------------

# Native GEE Köppen-Geiger present-day classification. Confirm the exact
# asset id in the catalog; a widely-used one is the Beck et al. (2023)
# present-day map. The band carries integer climate-class codes.
# Confirmed ingested asset (Beck et al. 2023, present-day 1991-2020,
# 1km classification). Single band 'b1', integer class codes 1-30.
# Kenya class histogram confirmed: codes 1,2,3,4,6,7,8,9,11,12,15,16,29
# all in-range, arid steppe (BSh=6) dominant - correct for Kenya.
KOPPEN_ASSET = "projects/thesis-project-502010/assets/koppen_present_1km"
KOPPEN_BAND = "b1"


def build_dominant_climate(
    polygons: "ee.FeatureCollection",
    koppen_image_id: str,
    id_property: str = "FNID",
    band: str = None,
) -> "ee.FeatureCollection":
    """
    Dominant (modal) Köppen-Geiger class per unit. The classification is
    a categorical raster of integer climate-class codes; the mode
    reducer returns the most common code within each polygon.
    """
    kg = ee.Image(koppen_image_id)
    if band is not None:
        kg = kg.select(band)

    def reduce_unit(f):
        mode = kg.reduceRegion(
            reducer=ee.Reducer.mode(), geometry=f.geometry(),
            scale=1000, maxPixels=1e9, bestEffort=True,
        ).values().get(0)
        return ee.Feature(None, {
            id_property: f.get(id_property),
            "climate_class": mode,
        })

    return polygons.map(reduce_unit)


# ---------------------------------------------------------------------
# Taxonomy-consistency check (D21 s648: confirm shared class set)
# ---------------------------------------------------------------------

def collect_climate_classes(panel) -> set:
    """
    Set of climate-class codes that are the DOMINANT (modal) class of at
    least one unit - i.e. the codes that actually enter the feature after
    mode-reduction. This is deliberately the unit-level modal set, NOT the
    raw pixel histogram: a class covering a handful of boundary pixels
    (e.g. Kenya codes 7,11 at 4-5 pixels) is never any unit's mode and so
    is not a feature level. The cross-country taxonomy check (D21 s648)
    must compare these modal sets, because a trace pixel-class present in
    one country's raster but dominant in another's units would otherwise
    look like a shared level when it is really a per-country asymmetry.
    """
    import pandas as pd
    return set(pd.to_numeric(panel["climate_class"], errors="coerce")
               .dropna().astype(int).unique())


# Valid Köppen-Geiger class codes (Beck et al. 2023 present-day scheme:
# 30 classes, codes 1-30). Adjust to the exact scheme of the chosen
# asset if it differs. Checking against a KNOWN valid set - not a self-
# referential union - is what makes the check able to FAIL.
KOPPEN_VALID_CODES = set(range(1, 31))


def assert_shared_taxonomy(class_sets: dict,
                           valid_codes: set = KOPPEN_VALID_CODES) -> None:
    """
    Confirm the three countries' climate classes are all valid codes in
    the KNOWN Köppen-Geiger taxonomy, and that the countries OVERLAP
    (share levels) rather than partitioning into disjoint schemes.

    This checks against an external valid-code set, not a self-referential
    union, so it can actually fail: a country returning codes outside the
    Köppen range (a different raster, a nodata sentinel like 0 or 255, a
    scaling error) is caught, and countries with zero shared classes
    (which would mean the categorical levels are NOT shared across the
    feature vector) are caught.
    """
    print("Köppen-Geiger classes per country:")
    for country, s in class_sets.items():
        print(f"  {country:<10}: {sorted(s)}")

    # 1. every class must be a valid Köppen code
    for country, s in class_sets.items():
        invalid = s - valid_codes
        if invalid:
            raise RuntimeError(
                f"{country} has codes {sorted(invalid)} outside the known "
                f"Köppen-Geiger taxonomy {min(valid_codes)}-{max(valid_codes)}"
                f" - wrong raster, nodata sentinel, or scaling error."
            )

    # 2. countries must share levels (overlap), not partition disjointly
    all_sets = list(class_sets.values())
    shared = set.intersection(*all_sets) if all_sets else set()
    union = set().union(*all_sets) if all_sets else set()
    print(f"Union of classes present : {sorted(union)}")
    print(f"Shared across all three  : {sorted(shared)}")
    if not shared:
        raise RuntimeError(
            "No climate class is shared across all three countries. The "
            "categorical levels would be effectively country-specific, "
            "breaking feature-vector integrity (D21 §648)."
        )

    print("\u2713 All classes are valid Köppen codes and the countries "
          "share levels; categorical is consistent across the feature vector.")
