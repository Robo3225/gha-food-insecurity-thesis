"""
============================================================

calendar.py

Date utilities for monthly feature extraction.

Author:
Robert Odingo MSc Thesis

============================================================
"""

from datetime import date


def generate_months(
    start_year,
    start_month,
    end_year,
    end_month,
):
    """
    Generate inclusive list of (year, month).

    Example

    generate_months(2020,1,2020,4)

    returns

    [(2020,1),
     (2020,2),
     (2020,3),
     (2020,4)]
    """

    start = date(start_year, start_month, 1)
    end = date(end_year, end_month, 1)

    months = []

    current = start

    while current <= end:

        months.append(
            (
                current.year,
                current.month,
            )
        )

        if current.month == 12:

            current = date(
                current.year + 1,
                1,
                1,
            )

        else:

            current = date(
                current.year,
                current.month + 1,
                1,
            )

    return months