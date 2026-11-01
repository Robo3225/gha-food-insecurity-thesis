"""
============================================================

build_dataset.py

Generic dataset builder.

This script orchestrates feature extraction,
merges all extracted features into a single
modelling dataset, and saves the result.

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

from scripts.gee.engine.extract_time_series import (
    extract_time_series,
)

from scripts.gee.metadata.write_manifest import (
    write_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "datasets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def build_dataset(
    country,
    features,
    start_year,
    start_month,
    end_year,
    end_month,
):
    """
    Build a modelling dataset.
    """

    print("=" * 70)
    print(f"Building dataset : {country['name']}")
    print("=" * 70)

    polygons = load_feature_collection(
        country["asset"]
    )

    dataset = None

    for feature in features:

        print()
        print("-" * 70)
        print(f"Extracting feature : {feature['name']}")
        print("-" * 70)

        df = extract_time_series(

            polygons=polygons,

            feature=feature,

            expected_rows=country["fsc_polygons"],

            start_year=start_year,
            start_month=start_month,

            end_year=end_year,
            end_month=end_month,

        )

        if dataset is None:

            dataset = df

        else:

            dataset = dataset.merge(

                df,

                on=[
                    "FNID",
                    "year",
                    "month",
                ],

                how="inner",

            )

    return dataset


def save_dataset(
    dataframe,
    filename,
):
    """
    Save dataset to CSV.
    """

    output_file = OUTPUT_DIR / filename

    dataframe.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 70)
    print("DATASET SAVED")
    print("=" * 70)
    print(output_file)

    return output_file


def main():

    from scripts.gee.features.ndvi import FEATURE as NDVI

    initialize(PROJECT_ID)

    COUNTRY = KENYA

    FEATURES = [

        NDVI,

    ]

    START_YEAR = 2020
    START_MONTH = 1

    END_YEAR = 2020
    END_MONTH = 3

    dataset = build_dataset(

        country=COUNTRY,

        features=FEATURES,

        start_year=START_YEAR,
        start_month=START_MONTH,

        end_year=END_YEAR,
        end_month=END_MONTH,

    )

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(dataset.head())

    print()

    print(f"Rows    : {len(dataset):,}")
    print(f"Columns : {len(dataset.columns)}")

    output_file = save_dataset(

        dataframe=dataset,

        filename="kenya_dataset_test.csv",

    )

    write_manifest(

        output_file=output_file,

        country=COUNTRY,

        features=FEATURES,

        start_year=START_YEAR,
        start_month=START_MONTH,

        end_year=END_YEAR,
        end_month=END_MONTH,

        dataframe=dataset,

    )

    print()
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()