"""
============================================================

test_ndvi_timeseries.py

Purpose
-------
Validate the generic time-series extraction engine
using three months of MODIS NDVI.

Expected rows:
640 reporting units × 3 months = 1920 rows

Author:
Robert Odingo MSc Thesis

============================================================
"""

from pathlib import Path

from config.config import PROJECT_ID
from config.countries import KENYA

from scripts.gee.extractors.gee_utils import (
    initialize,
    load_feature_collection,
)

from scripts.gee.engine.extract_time_series import (
    extract_time_series,
)

from scripts.gee.features.ndvi import FEATURE


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "ndvi_test_timeseries.csv"


def main():

    print("=" * 70)
    print("NDVI TIME SERIES TEST")
    print("=" * 70)

    initialize(PROJECT_ID)

    polygons = load_feature_collection(KENYA["asset"])

    df = extract_time_series(

        polygons=polygons,

        feature=FEATURE,

        expected_rows=KENYA["fsc_polygons"],

        start_year=2020,
        start_month=1,

        end_year=2020,
        end_month=3,

    )

    print()

    print(df.head())

    print()

    print(f"Rows extracted : {len(df)}")

    expected = 640 * 3

    print(f"Expected rows  : {expected}")

    if len(df) == expected:

        print("\n✓ Row count validation PASSED")

    else:

        print("\n✗ Row count validation FAILED")

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nSaved:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()