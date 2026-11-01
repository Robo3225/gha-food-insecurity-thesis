"""
======================================================================

canonicalize.py

Purpose
-------
Single source of truth for administrative-name canonicalization
across the FNID <-> admin_code crosswalk pipeline.

Both the FEWS standardizer and the FSC validator import from
this module so that the two sides of the crosswalk merge are
canonicalized IDENTICALLY. Any divergence between the two sides
is the single most common cause of a silent 0% match, so the
logic lives here once and only here.

Two operations are provided and MUST be applied in this order:

1. canonicalize_text(text)
   Lowercase, normalize punctuation/whitespace. Applied to every
   administrative name field on both sides.

2. normalize_admin_names(df, columns)
   Apply per-level naming-convention rules (e.g. FEWS "snnp" ->
   canonical "snnpr"). Applied AFTER the fields are split into
   ADMIN1/ADMIN2/ADMIN3 columns and stripped.

Design notes
------------
- These functions are pure and side-effect free. normalize_admin_names
  returns a copy and never mutates its argument.
- The naming-convention rules are keyed on the CANONICAL (post
  canonicalize_text) form. That is why "benishangul gumz" appears
  lowercase with no punctuation here.
- Rules map the FEWS spelling to the FSC spelling, because the FSC
  shapefile hierarchy is treated as the reference geography
  (World Bank harmonized dataset is keyed to it).

Author:
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------
# Canonical administrative hierarchy column names
# ---------------------------------------------------------------------

HIERARCHY = ["ADMIN1", "ADMIN2", "ADMIN3"]


# ---------------------------------------------------------------------
# Text-level canonicalization
# ---------------------------------------------------------------------
#
# Punctuation replacement rules applied to raw administrative names.
# Note that "," is NOT mapped here: comma splitting of the composite
# admin_name string is the caller's responsibility (the FEWS side
# splits a single "a, b, c, country" string; the FSC side already
# has separate columns). Keeping comma handling out of this function
# means both sides share exactly the same per-field cleaning.

_PUNCTUATION_REPLACEMENTS = {
    "\r": " ",
    "\n": " ",
    "/": " ",
    "-": " ",
    "'": "",
    "\u2019": "",   # right single quotation mark
}


def canonicalize_text(text) -> str:
    """
    Convert a single administrative name field into canonical form:
    lowercased, punctuation normalized, whitespace collapsed, and
    leading/trailing whitespace stripped.

    Returns an empty string for missing values.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    for old, new in _PUNCTUATION_REPLACEMENTS.items():
        text = text.replace(old, new)

    # Collapse repeated whitespace and strip leading/trailing space.
    # The strip is critical: without it, values produced by splitting
    # a composite string on a delimiter retain the space that followed
    # the delimiter (" snnp" != "snnp"), which silently breaks both
    # the hierarchy merge AND any exact-match normalization rule.
    text = " ".join(text.split())

    return text


# ---------------------------------------------------------------------
# Administrative naming-convention normalization
# ---------------------------------------------------------------------
#
# Keyed on the CANONICAL (post canonicalize_text) form.
# Maps the FEWS spelling -> the FSC (reference) spelling.
#
# Verified Ethiopia divergences:
#   snnp             -> snnpr
#   benishangul gumz -> benishangul gumuz
#
# Add Kenya / Somalia divergences here as they are discovered so
# that all three countries share one rule table.

ADMIN_NORMALIZATION = {
    "ADMIN1": {
        "snnp": "snnpr",
        "benishangul gumz": "benishangul gumuz",
    },
    "ADMIN2": {},
    "ADMIN3": {},
}


def normalize_admin_names(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply per-level naming-convention rules to already-canonicalized
    hierarchy columns. Returns a copy; does not mutate the input.

    columns defaults to HIERARCHY. Only columns present in both the
    DataFrame and ADMIN_NORMALIZATION (with non-empty rules) are touched.
    """

    if columns is None:
        columns = HIERARCHY

    df = df.copy()

    for column in columns:

        rules = ADMIN_NORMALIZATION.get(column)

        if column in df.columns and rules:
            df[column] = df[column].replace(rules)

    return df


# ---------------------------------------------------------------------
# Convenience: canonicalize a set of already-separated columns
# ---------------------------------------------------------------------

def canonicalize_hierarchy(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply canonicalize_text then normalize_admin_names to a DataFrame
    that ALREADY has separate ADMIN1/ADMIN2/ADMIN3 columns (i.e. the
    FSC side, or the FEWS side after splitting admin_name).

    Returns a copy.
    """

    if columns is None:
        columns = HIERARCHY

    df = df.copy()

    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(canonicalize_text)

    df = normalize_admin_names(df, columns)

    return df


# ---------------------------------------------------------------------
# Invariant check
# ---------------------------------------------------------------------

def assert_canonical(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> None:
    """
    Assert that the given columns are already in canonical form:
    no leading/trailing whitespace, no uppercase. Raises AssertionError
    with the offending column and sample values otherwise.

    Call this immediately after canonicalization to turn silent
    corruption into a loud failure.
    """

    if columns is None:
        columns = HIERARCHY

    for column in columns:

        if column not in df.columns:
            continue

        col = df[column].astype(str)

        stripped = col == col.str.strip()
        if not stripped.all():
            bad = col[~stripped].unique()[:5]
            raise AssertionError(
                f"Column '{column}' has values with surrounding "
                f"whitespace, e.g. {list(bad)!r}"
            )

        lowered = col == col.str.lower()
        if not lowered.all():
            bad = col[~lowered].unique()[:5]
            raise AssertionError(
                f"Column '{column}' has non-lowercase values, "
                f"e.g. {list(bad)!r}"
            )

