"""
======================================================================

export_composites.py

Purpose
-------
Submit a batch export of composite-resolution feature values to Google
Drive, one row per reporting unit per source composite across the full
window. This is the GEE half of the composite-resolution extraction
design; aggregate_composites.py is the local half.

Why batch export rather than the getInfo() loop
-----------------------------------------------
The client-side path (featurecollection_to_list -> getInfo) transfers
results synchronously and is capped near 5000 features with a
computation timeout. At 639 units x ~164 composites (72 months of
16-day MOD13Q1) a single getInfo would be ~105,000 rows - far past the
cap. ee.batch.Export.table.toDrive runs asynchronously on GEE's batch
backend, which has no such cap, and writes a CSV to Drive.

Design
------
For each source composite in the window we reduce the (masked,
scaled-at-export-time? NO) image over the units and tag each output
feature with the composite's start date. Scaling/offset is applied
LOCALLY after download, matching extract_feature's behaviour, so the
raw reducer output is exported unchanged and the scale factor lives in
one place (the feature dict) rather than being baked into the export.

The feature's existing `aggregation` function is NOT used here: that
function collapses a month to one image. For composite-resolution
export we instead map the feature's QUALITY MASK over each composite
and reduce each one separately. The feature dict therefore must expose
its masking via `apply_quality_mask` (already defined in ndvi.py /
lst.py); chirps has none and is handled as a daily-sum special case
elsewhere.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import ee


def build_composite_export(
    polygons: "ee.FeatureCollection",
    feature: dict,
    start: str,       # "YYYY-MM-DD"
    end: str,         # "YYYY-MM-DD" exclusive
    mask_fn=None,     # callable(ee.Image)->ee.Image, or None
    id_property: str = "FNID",
    description: str = None,
    folder: str = "gee_exports",
) -> "ee.batch.Task":
    """
    Build and RETURN (does not start) an export task that writes
    composite-resolution zonal reductions to Drive.

    The returned task must be .start()ed by the caller, so submission is
    explicit and inspectable.

    Output CSV columns (per row): id_property, composite_date,
    <reducer_output>. Scaling is applied locally after download.
    """

    band = feature["bands"][0]          # the signal band, e.g. "NDVI"
    reducer_output = feature["reducer_output"]
    scale = feature["scale"]
    tile_scale = feature.get("tile_scale", 4)

    collection = (
        ee.ImageCollection(feature["collection"])
        .filterDate(start, end)
        .select(feature["bands"])       # signal + QA bands
    )

    def reduce_one(image):
        # Apply the feature's quality mask if provided, then keep only
        # the signal band for reduction.
        img = mask_fn(image) if mask_fn is not None else image
        img = img.select(band)

        date = image.date().format("YYYY-MM-dd")

        reduced = img.reduceRegions(
            collection=polygons,
            reducer=feature["reducer"](),
            scale=scale,
            tileScale=tile_scale,
        )

        # Tag every output feature with the composite date and drop
        # geometry so the export is a compact table.
        def tag(f):
            return ee.Feature(None, {
                id_property: f.get(id_property),
                "composite_date": date,
                reducer_output: f.get(reducer_output),
            })

        return reduced.map(tag)

    # flatten: one FeatureCollection per composite -> a single table
    exported = ee.FeatureCollection(
        collection.map(reduce_one)
    ).flatten()

    description = description or f"{feature['name']}_composites"

    task = ee.batch.Export.table.toDrive(
        collection=exported,
        description=description,
        folder=folder,
        fileFormat="CSV",
        selectors=[id_property, "composite_date", reducer_output],
    )

    return task
