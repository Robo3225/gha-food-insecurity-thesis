"""
======================================================================

build_conflict_features.py  (Config D — ACLED conflict feature, D27)

Standalone conflict-feature artifact for the Config-D ablation. NOT merged
into the frozen v2 panels (D27: ablation arm, no re-freeze) — produced as
a per-unit-month table joined at the harness for Config D only.

Feature construction (D27 pre-registration):
  - Source: ACLED event-level (Research tier) — events with lat/lon,
    event_date, fatalities, for Kenya/Ethiopia/Somalia.
  - Spatial: 50 km GAUSSIAN kernel smoothing. Each reporting unit's
    conflict exposure at month m = sum over events in month m of
    w(d) * intensity, where d = distance(event, unit centroid) and
    w(d) = exp(-d^2 / (2*sigma^2)), sigma chosen so the kernel ~ 50 km
    (sigma = 50/ (2.something) -> here sigma=50km, truncated at ~2.5 sigma).
  - Two features:
      conflict_density   = Gaussian-weighted EVENT count (exposure)
      conflict_intensity = Gaussian-weighted FATALITY sum (severity)
  - Temporal: monthly, 2018-01..2023-12 (matches panel window).
  - Leakage-safe: a unit-month's value uses only events IN that month
    (no future); lags (if used in Config D) built within-unit backward,
    same discipline as biophysical features (D19).

Output: data/processed/conflict/{country}_conflict_features.csv
  columns: canonical_admin_code, year_month, conflict_density,
           conflict_intensity

Runs in gee_env (needs the ACLED export + unit centroids/geometries).
This module is the extraction/smoothing spec + implementation; the
harness join for Config D is in model_configD.py.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 50 km Gaussian kernel. sigma in km; weight = exp(-d^2/(2 sigma^2)).
# truncate at 2.5 sigma (~125 km) for efficiency; beyond that weight ~0.
SIGMA_KM = 50.0
TRUNCATE_KM = 2.5 * SIGMA_KM
EARTH_R_KM = 6371.0

COUNTRY_ISO = {"kenya": "KEN", "ethiopia": "ETH", "somalia": "SOM"}


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (km) between one point and arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def _gaussian_w(d_km):
    return np.exp(-(d_km**2) / (2 * SIGMA_KM**2))


def build_country(country, acled_csv, centroids_csv, out_dir):
    """
    country: slug. acled_csv: ACLED export (event_date, latitude, longitude,
    fatalities, country). centroids_csv: per-unit centroid
    (canonical_admin_code, cent_lat, cent_lon).
    """
    iso = COUNTRY_ISO[country]
    ac = pd.read_csv(acled_csv)
    # normalise expected ACLED columns
    ac = ac.rename(columns={c: c.lower() for c in ac.columns})
    ac = ac[ac["country"].str.lower() == country]  # ACLED uses country name
    ac["event_date"] = pd.to_datetime(ac["event_date"])
    ac = ac[(ac["event_date"] >= "2018-01-01") & (ac["event_date"] <= "2023-12-31")]
    ac["year_month"] = ac["event_date"].dt.strftime("%Y_%m")
    ac["fatalities"] = pd.to_numeric(ac["fatalities"], errors="coerce").fillna(0)

    cent = pd.read_csv(centroids_csv)  # canonical_admin_code, cent_lat, cent_lon

    months = pd.period_range("2018-01", "2023-12", freq="M").strftime("%Y_%m")
    rows = []
    for ym in months:
        ev = ac[ac["year_month"] == ym]
        if len(ev) == 0:
            for _, u in cent.iterrows():
                rows.append({"canonical_admin_code": u["canonical_admin_code"],
                             "year_month": ym, "conflict_density": 0.0,
                             "conflict_intensity": 0.0})
            continue
        ev_lat = ev["latitude"].to_numpy(); ev_lon = ev["longitude"].to_numpy()
        ev_fat = ev["fatalities"].to_numpy()
        for _, u in cent.iterrows():
            d = _haversine_km(u["cent_lat"], u["cent_lon"], ev_lat, ev_lon)
            near = d <= TRUNCATE_KM
            if not near.any():
                dens = inten = 0.0
            else:
                w = _gaussian_w(d[near])
                dens = float(w.sum())                      # weighted event count
                inten = float((w * ev_fat[near]).sum())    # weighted fatalities
            rows.append({"canonical_admin_code": u["canonical_admin_code"],
                         "year_month": ym, "conflict_density": dens,
                         "conflict_intensity": inten})

    out = pd.DataFrame(rows)
    outp = Path(out_dir) / f"{country}_conflict_features.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False)
    print(f"{country}: {len(out):,} unit-months -> {outp}")
    print(f"  density  range: {out.conflict_density.min():.2f}..{out.conflict_density.max():.2f}")
    print(f"  intensity range: {out.conflict_intensity.min():.2f}..{out.conflict_intensity.max():.2f}")
    nz = (out.conflict_density > 0).mean()
    print(f"  nonzero-exposure unit-months: {nz:.1%}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    ap.add_argument("--acled", required=True, help="ACLED export CSV")
    ap.add_argument("--centroids", required=True, help="per-unit centroid CSV")
    ap.add_argument("--out", default="data/processed/conflict")
    args = ap.parse_args()
    build_country(args.country, args.acled, args.centroids, args.out)


if __name__ == "__main__":
    main()
