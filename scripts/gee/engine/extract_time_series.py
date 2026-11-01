"""
============================================================

extract_time_series.py

Generic monthly time-series extraction engine.

Loops over a sequence of months, extracts a feature
for each month, validates the output, and returns
a single concatenated DataFrame.

Author:
Robert Odingo MSc Thesis

============================================================
"""

import pandas as pd

from scripts.gee.engine.extract_feature import extract_feature
from scripts.gee.utils.calendar import generate_months


def extract_time_series(
    polygons,
    feature,
    expected_rows,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
):
    """
    Extract a monthly time series for a single feature.

    Parameters
    ----------
    polygons : ee.FeatureCollection
        Reporting units.

    feature : dict
        Feature configuration.

    expected_rows : int
        Expected number of reporting units.

    start_year : int
    start_month : int
    end_year : int
    end_month : int

    Returns
    -------
    pandas.DataFrame
    """

    months = generate_months(
        start_year,
        start_month,
        end_year,
        end_month,
    )

    total_months = len(months)

    all_data = []

    print("=" * 60)
    print(f"Feature : {feature['name']}")
    print(f"Months  : {total_months}")
    print(f"Expected reporting units : {expected_rows}")
    print("=" * 60)

    for i, (year, month) in enumerate(months, start=1):

        print(f"[{i}/{total_months}] {year}-{month:02d}")

        df = extract_feature(
            polygons=polygons,
            feature=feature,
            year=year,
            month=month,
        )

        # --------------------------------------------------
        # Validate extraction
        # --------------------------------------------------

        if len(df) != expected_rows:

            raise RuntimeError(
                f"{year}-{month:02d}: "
                f"Expected {expected_rows} reporting units "
                f"but received {len(df)}."
            )

        all_data.append(df)

    print("\nConcatenating monthly data...")

    final_df = pd.concat(
        all_data,
        ignore_index=True,
    )

    print(f"Total rows extracted: {len(final_df)}")

    return final_df