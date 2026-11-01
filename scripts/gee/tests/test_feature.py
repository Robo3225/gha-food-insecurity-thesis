"""
============================================================

test_feature.py

Purpose
-------
Test the generic Google Earth Engine feature
extraction framework by extracting a single
environmental feature for one month over the
Kenya FEWS reporting units.

Author:
Robert Odingo MSc Thesis

============================================================
"""

from pathlib import Path

import pandas as pd

from config.config import PROJECT_ID 
from config.countries import KENYA

from scripts.gee.extractors.gee_utils import (
    initialize,
    load_feature_collection,
)

from scripts.gee.engine.extract_feature import (
    extract_feature,
)

from scripts.gee.features.lst import FEATURE


# ==========================================================
# Configuration
# ==========================================================

YEAR = 2020
MONTH = 1

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = None


# ==========================================================
# Main
# ==========================================================

def main():

    feature_name = FEATURE.get("description", FEATURE["name"])
    feature_name = FEATURE["name"]

    print("=" * 70)
    print(f"{FEATURE['name'].upper()} PIPELINE TEST")
    print("=" * 70)

    initialize(PROJECT_ID)

    OUTPUT_FILE = (OUTPUT_DIR / f"{FEATURE['output']}_test_{YEAR}_{MONTH:02d}.csv")

    print("\nLoading Kenya FSC reporting units...")

    polygons = load_feature_collection( KENYA["asset"] )

    print(f"\nExtracting {FEATURE['name'].upper()}...")

    df = extract_feature(
        polygons=polygons,
        feature=FEATURE,
        year=YEAR,
        month=MONTH,
    )

    print("\nPreview")

    print(df.head())

    print(f"\nReporting units: {len(df)}")

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved:\n{OUTPUT_FILE}")

    print("\nSUCCESS")


if __name__ == "__main__":
    main()