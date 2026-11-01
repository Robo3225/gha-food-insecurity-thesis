"""
======================================================================

export_unit_centroids.py  (Config D prerequisite)

Emits per-reporting-unit centroids (canonical_admin_code, cent_lat,
cent_lon) for the 50km Gaussian conflict-smoothing (build_conflict_features).

Reads the FSC / reporting-unit geometries already used for spatial
validation and the crosswalk, computes each unit's centroid, and keys it
on canonical_admin_code via the existing FNID->admin_code crosswalk — so
the centroid table joins directly to the conflict features and the panels.

Centroid note: computed in an equal-area projection then converted back to
lat/lon, so the centroid is the true geometric centroid (not distorted by
computing in degrees). For the Gaussian kernel a small centroid error is
negligible (50km sigma), but doing it right costs nothing.

Usage:
  python -m scripts.06_feature_extraction.export_unit_centroids --country somalia

Output: data/processed/metadata/{country}_centroids.csv

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# equal-area CRS for centroid computation (Africa Albers); then back to WGS84
EQUAL_AREA_CRS = "ESRI:102022"   # Africa Albers Equal Area Conic
WGS84 = "EPSG:4326"


def _find_geometry_file(country, root):
    """Locate the FSC / reporting-unit shapefile or geojson for the country.
    Adjust the candidates to the real repo layout if needed."""
    slug = country.lower()
    candidates = [
        root / "data" / "raw" / "shapefiles" / f"{slug}_fsc.shp",
        root / "data" / "raw" / "shapefiles" / f"{slug}.shp",
        root / "data" / "raw" / "shapefiles" / slug / f"{slug}.shp",
        root / "data" / "raw" / "fsc" / f"{slug}.geojson",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise RuntimeError(
        f"No geometry file found for {country}. Looked in: "
        + "; ".join(str(c) for c in candidates)
        + ". Point --geometry at the real file."
    )


def export_centroids(country, root, geometry_path=None, id_field=None):
    import geopandas as gpd

    root = Path(root)
    gpath = Path(geometry_path) if geometry_path else _find_geometry_file(country, root)
    gdf = gpd.read_file(gpath)
    print(f"{country}: read {len(gdf)} geometries from {gpath}")

    # identify the unit id field (FNID or admin_code, depending on the source)
    if id_field is None:
        for cand in ["FNID", "fnid", "admin_code", "ADMIN_CODE", "canonical_admin_code"]:
            if cand in gdf.columns:
                id_field = cand; break
    if id_field is None:
        raise RuntimeError(f"No id field found in {list(gdf.columns)}; pass --id-field")
    print(f"  unit id field: {id_field}")

    # true centroid via equal-area projection, then back to lat/lon
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    cent = gdf.to_crs(EQUAL_AREA_CRS).geometry.centroid.to_crs(WGS84)

    out = pd.DataFrame({
        id_field: gdf[id_field].values,
        "cent_lon": cent.x.values,
        "cent_lat": cent.y.values,
    })

    # key on canonical_admin_code: if the geometry is FNID-keyed, join the crosswalk
    if id_field.upper() == "FNID":
        cw = pd.read_csv(root / "data" / "crosswalks" /
                         f"{country.lower()}_fnid_admin_code_crosswalk.csv")
        out = out.merge(cw[["FNID", "admin_code"]], left_on=id_field, right_on="FNID",
                        how="left")
        unmatched = out["admin_code"].isna().sum()
        if unmatched:
            print(f"  WARNING: {unmatched} FNIDs unmatched in crosswalk")
        out = out.rename(columns={"admin_code": "canonical_admin_code"})
    else:
        out = out.rename(columns={id_field: "canonical_admin_code"})

    out = out[["canonical_admin_code", "cent_lat", "cent_lon"]].dropna(
        subset=["canonical_admin_code"])

    outp = root / "data" / "processed" / "metadata" / f"{country.lower()}_centroids.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False)
    print(f"  -> {len(out)} centroids -> {outp}")
    print(f"  lat range {out.cent_lat.min():.2f}..{out.cent_lat.max():.2f}, "
          f"lon range {out.cent_lon.min():.2f}..{out.cent_lon.max():.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--geometry", default=None, help="explicit path to unit geometry")
    ap.add_argument("--id-field", default=None, help="unit id column (default: auto-detect)")
    args = ap.parse_args()
    export_centroids(args.country, args.root, args.geometry, args.id_field)


if __name__ == "__main__":
    main()
