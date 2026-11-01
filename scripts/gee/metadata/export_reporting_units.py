"""
============================================================

export_reporting_units.py

Exports reporting-unit metadata from an Earth Engine
FeatureCollection.

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
    featurecollection_to_list,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metadata"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def export_reporting_units(country):

    print("=" * 60)
    print(country["name"])
    print("=" * 60)

    polygons = load_feature_collection(
        country["asset"]
    )

    rows = featurecollection_to_list(polygons)

    df = pd.DataFrame(rows)

    keep = [
        c
        for c in [
            "FNID",
            "ADMIN0",
            "ADMIN1",
            "ADMIN2",
            "ADMIN3",
            "LZCODE",
            "LZNAME",
        ]
        if c in df.columns
    ]

    df = (
        df[keep]
        .sort_values("FNID")
        .reset_index(drop=True)
    )

    output = (
        OUTPUT_DIR
        / f"{country['code'].lower()}_reporting_units.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print()

    print(df.head())

    print()

    print(f"Rows exported: {len(df)}")

    print(f"\nSaved:\n{output}")


def main():

    initialize(PROJECT_ID)

    export_reporting_units(KENYA)


if __name__ == "__main__":
    main()