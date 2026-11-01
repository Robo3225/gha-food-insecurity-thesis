"""
============================================================

extract_feature.py

Generic feature extraction engine.

Extracts a single environmental feature for one month,
applies feature-specific aggregation and zonal reduction,
validates the output, and returns a standardized DataFrame.

Standard output schema

FNID
year
month
<feature>

Author:
Robert Odingo MSc Thesis

============================================================
"""

from __future__ import annotations

import ee
import pandas as pd

from scripts.gee.extractors.gee_utils import (
    monthly_dates,
    zonal_reduce,
    featurecollection_to_list,
)

from scripts.gee.validators.feature_validator import (
    validate_feature,
)


# ----------------------------------------------------------
# Main extraction function
# ----------------------------------------------------------

def extract_feature(
    polygons: ee.FeatureCollection,
    feature: dict,
    year: int,
    month: int,
) -> pd.DataFrame:
    """
    Extract one environmental feature for one month.

    Parameters
    ----------
    polygons : ee.FeatureCollection
        Reporting-unit polygons.

    feature : dict
        Environmental feature specification.

    year : int
        Extraction year.

    month : int
        Extraction month.

    Returns
    -------
    pandas.DataFrame
        Standardized feature table with columns
        FNID, year, month, <feature output>.

    Notes
    -----
    Units with no valid pixels (e.g. fully cloud-masked)
    are retained with null values rather than dropped.
    Missing data is reported but never silently removed.
    """

    # ------------------------------------------------------
    # Validate feature definition
    # ------------------------------------------------------

    validate_feature(feature)

    print(
        f"\nExtracting {feature['name']} "
        f"({year}-{month:02d})"
    )

    # ------------------------------------------------------
    # Monthly dates
    # ------------------------------------------------------

    start, end = monthly_dates(year, month)

    # ------------------------------------------------------
    # Load image collection
    # ------------------------------------------------------

    collection = (
        ee.ImageCollection(feature["collection"])
        .filterDate(start, end)
        .select(feature["bands"])
    )

    image_count = collection.size().getInfo()

    print(f"✓ Images found: {image_count}")

    if image_count == 0:
        raise RuntimeError(
            f"No images found for "
            f"{feature['name']} "
            f"{year}-{month:02d}"
        )

    # ------------------------------------------------------
    # Feature-specific aggregation
    # ------------------------------------------------------

    image = feature["aggregation"](collection)

    # ------------------------------------------------------
    # Zonal reduction
    # ------------------------------------------------------

    results = zonal_reduce(
        image=image,
        polygons=polygons,
        reducer=feature["reducer"](),
        scale=feature["scale"],
        tile_scale=feature.get("tile_scale", 4),
    )

    # ------------------------------------------------------
    # Convert FeatureCollection to DataFrame
    # ------------------------------------------------------

    rows = featurecollection_to_list(results)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "Earth Engine returned an empty DataFrame."
        )

    if "FNID" not in df.columns:
        raise RuntimeError(
            "FNID column missing from Earth Engine output."
        )

    # ------------------------------------------------------
    # Locate the reducer output column
    # ------------------------------------------------------
    #
    # reduceRegions names its output after the reducer:
    #   ee.Reducer.mean()   -> "mean"
    #   ee.Reducer.sum()    -> "sum"
    #   ee.Reducer.median() -> "median"
    #
    # The expected column name is declared explicitly in
    # the feature definition via "reducer_output".
    #
    # If a polygon contains no valid pixels, reduceRegions
    # omits the property entirely rather than returning
    # null. When this happens for EVERY polygon in a month,
    # the column is absent from the DataFrame. This is a
    # data-quality condition, not a failure, so the column
    # is created and filled with nulls for downstream
    # handling.

    reducer_col = feature["reducer_output"]

    if reducer_col not in df.columns:
        print(
            f"⚠ Warning: no valid pixels for ANY reporting "
            f"unit in {year}-{month:02d}. "
            f"All values set to null."
        )
        df[reducer_col] = float("nan")

    # ------------------------------------------------------
    # Apply scaling and offset
    # ------------------------------------------------------
    #
    # Some products require an additive offset in addition
    # to a multiplicative scale factor. For example, MODIS
    # land surface temperature is stored in Kelvin with a
    # scale factor of 0.02; conversion to Celsius requires
    # an offset of -273.15.
    #
    # Features without an offset default to zero, leaving
    # existing behaviour unchanged.

    offset = feature.get("offset", 0.0)

    df[feature["output"]] = (
        df[reducer_col] * feature["multiplier"] + offset
    )

    # ------------------------------------------------------
    # Missing-value diagnostics
    # ------------------------------------------------------

    n_null = df[feature["output"]].isna().sum()
    n_total = len(df)

    if n_null > 0:
        pct = 100.0 * n_null / n_total
        print(
            f"⚠ Warning: {n_null}/{n_total} reporting units "
            f"({pct:.1f}%) returned no valid pixels."
        )

    # ------------------------------------------------------
    # Standard output schema
    # ------------------------------------------------------

    df["year"] = year
    df["month"] = month

    df = df[
        [
            "FNID",
            "year",
            "month",
            feature["output"],
        ]
    ]

    print(
        f"✓ Extraction complete "
        f"({len(df)} reporting units)"
    )

    return df