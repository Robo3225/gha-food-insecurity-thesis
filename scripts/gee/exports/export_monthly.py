"""
======================================================================

export_monthly.py

Batch export of MONTHLY-aggregated feature values to Drive, one row
per reporting unit per month across a window.

Sibling to export_composites.py. The difference is the temporal step:

  export_composites : exports each source composite; monthly
                      aggregation happens locally (NDVI: mean of
                      16-day composites).

  export_monthly    : aggregates each month to ONE image INSIDE GEE
                      using the feature's own aggregation() function,
                      then zonally reduces and exports. Required for
                      features whose monthly aggregation is not a simple
                      mean of the source images:
                        - CHIRPS: temporal SUM of daily precipitation
                        - LST   : QC-masked temporal MEAN of 8-day images

The feature's aggregation() and reducer() are reused unchanged from the
feature definition, so the sum-then-mean (CHIRPS) and mask-then-mean
(LST) logic lives in one place (the feature dict), not here.

Scaling/offset is applied LOCALLY after download (as in extract_feature
and export_composites), so the raw reducer output is exported unchanged.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import ee


def _month_starts(start_year: int, end_year: int):
    """Yield (year, month) for every month in [start_year, end_year]."""
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            yield y, m


def build_monthly_export(
    polygons: "ee.FeatureCollection",
    feature: dict,
    start_year: int,
    end_year: int,
    id_property: str = "FNID",
    description: str = None,
    folder: str = "gee_exports",
) -> "ee.batch.Task":
    """
    Build and RETURN (does not start) a batch export of monthly
    aggregated feature values.

    For each month in [start_year, end_year], the month's images are
    aggregated to a single image via feature["aggregation"], zonally
    reduced via feature["reducer"](), tagged with year/month, and
    flattened into one exported table.

    Output CSV columns: id_property, year, month, <reducer_output>.
    """

    reducer_output = feature["reducer_output"]
    scale = feature["scale"]
    tile_scale = feature.get("tile_scale", 4)
    bands = feature["bands"]
    collection_id = feature["collection"]
    aggregation = feature["aggregation"]

    months = ee.List([
        {"year": y, "month": m}
        for (y, m) in _month_starts(start_year, end_year)
    ])

    def reduce_month(entry):
        entry = ee.Dictionary(entry)
        year = ee.Number(entry.get("year"))
        month = ee.Number(entry.get("month"))

        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, "month")

        monthly_images = (
            ee.ImageCollection(collection_id)
            .filterDate(start, end)
            .select(bands)
        )

        # Guard against months with NO source images (e.g. MODIS begins
        # 2000-02-18, so a 2000-01 baseline month is empty; or a sporadic
        # satellite gap mid-record). An empty collection aggregates to a
        # band-less image, which fails reduceRegions and kills the whole
        # export. When the month is empty, emit null-valued features for
        # every unit instead of crashing, so the export completes and the
        # gap is handled downstream (D11 discipline).
        count = monthly_images.size()

        def _reduce_nonempty():
            image = aggregation(monthly_images)
            return image.reduceRegions(
                collection=polygons,
                reducer=feature["reducer"](),
                scale=scale,
                tileScale=tile_scale,
            )

        def _null_features():
            # one feature per unit with a null reducer output
            return polygons.map(
                lambda f: f.set(reducer_output, None)
            )

        reduced = ee.FeatureCollection(
            ee.Algorithms.If(
                count.gt(0), _reduce_nonempty(), _null_features()
            )
        )

        def tag(f):
            return ee.Feature(None, {
                id_property: f.get(id_property),
                "year": year,
                "month": month,
                reducer_output: f.get(reducer_output),
            })

        return reduced.map(tag)

    exported = ee.FeatureCollection(months.map(reduce_month)).flatten()

    description = description or f"{feature['name']}_monthly"

    task = ee.batch.Export.table.toDrive(
        collection=exported,
        description=description,
        folder=folder,
        fileFormat="CSV",
        selectors=[id_property, "year", "month", reducer_output],
    )
    return task
