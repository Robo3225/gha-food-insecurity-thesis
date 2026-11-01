"""
============================================================

feature_validator.py

Validation utilities for environmental feature
definitions.

Ensures that every feature definition satisfies
the requirements of the generic extraction engine.

Author:
Robert Odingo MSc Thesis

============================================================
"""

from __future__ import annotations


# ----------------------------------------------------------
# Required configuration keys and their expected types
# ----------------------------------------------------------

REQUIRED_KEYS: dict[str, type | tuple] = {
    "name": str,
    "collection": str,
    "bands": list,
    "aggregation": object,
    "reducer": object,
    "reducer_output": str,
    "scale": int,
    "multiplier": (int, float),
    "output": str,
}


# ----------------------------------------------------------
# Optional configuration keys and their expected types
# ----------------------------------------------------------

OPTIONAL_KEYS: dict[str, type | tuple] = {
    "description": str,
    "offset": (int, float),
    "tile_scale": int,
}


# ----------------------------------------------------------
# Reducer output names recognised by the engine
# ----------------------------------------------------------
#
# reduceRegions names its output property after the
# reducer used. This set documents the values that are
# valid for "reducer_output" and guards against typos.

VALID_REDUCER_OUTPUTS: set[str] = {
    "mean",
    "sum",
    "median",
    "min",
    "max",
    "stdDev",
    "variance",
    "count",
    "first",
    "mode",
}


def validate_feature(feature: dict) -> None:
    """
    Validate a feature configuration.

    Parameters
    ----------
    feature : dict
        Environmental feature definition.

    Raises
    ------
    TypeError
        Feature is not a dictionary, or a key holds a
        value of an incorrect type.

    KeyError
        One or more required keys are missing.

    ValueError
        A key holds a structurally invalid value.
    """

    # ------------------------------------------------------
    # Structure
    # ------------------------------------------------------

    if not isinstance(feature, dict):
        raise TypeError(
            "Feature definition must be a dictionary."
        )

    # ------------------------------------------------------
    # Required keys present
    # ------------------------------------------------------

    missing = [
        key
        for key in REQUIRED_KEYS
        if key not in feature
    ]

    if missing:
        raise KeyError(
            f"Missing feature configuration keys: {missing}"
        )

    # ------------------------------------------------------
    # Required key types
    # ------------------------------------------------------
    #
    # "aggregation" and "reducer" are callables rather than
    # concrete classes; they are checked separately below.

    for key, expected_type in REQUIRED_KEYS.items():

        if key in ("aggregation", "reducer"):
            continue

        if not isinstance(feature[key], expected_type):
            raise TypeError(
                f"'{key}' must be of type "
                f"{expected_type}, got "
                f"{type(feature[key])}."
            )

    # ------------------------------------------------------
    # Optional key types
    # ------------------------------------------------------

    for key, expected_type in OPTIONAL_KEYS.items():

        if key not in feature:
            continue

        if not isinstance(feature[key], expected_type):
            raise TypeError(
                f"'{key}' must be of type "
                f"{expected_type}, got "
                f"{type(feature[key])}."
            )

    # ------------------------------------------------------
    # Unknown keys
    # ------------------------------------------------------
    #
    # Catches silent typos such as "multipler" that would
    # otherwise be ignored by the engine.

    known_keys = set(REQUIRED_KEYS) | set(OPTIONAL_KEYS)

    unknown = [
        key
        for key in feature
        if key not in known_keys
    ]

    if unknown:
        raise KeyError(
            f"Unknown feature configuration keys: "
            f"{unknown}. Known keys: "
            f"{sorted(known_keys)}"
        )

    # ------------------------------------------------------
    # Value constraints
    # ------------------------------------------------------

    if len(feature["bands"]) == 0:
        raise ValueError(
            "Feature must contain at least one band."
        )

    if not all(
        isinstance(band, str)
        for band in feature["bands"]
    ):
        raise TypeError(
            "'bands' must contain only strings."
        )

    if not callable(feature["aggregation"]):
        raise TypeError(
            "'aggregation' must be callable."
        )

    # ------------------------------------------------------
    # Reducer
    # ------------------------------------------------------
    #
    # Reducers are stored as callables (e.g.
    # lambda: ee.Reducer.mean()) so that feature
    # definitions can be imported before Earth Engine
    # has been initialized.

    if not callable(feature["reducer"]):
        raise TypeError(
            "'reducer' must be a callable returning an ee.Reducer."
        )

    if feature["scale"] <= 0:
        raise ValueError(
            "'scale' must be greater than zero."
        )

    # ------------------------------------------------------
    # Multiplier
    # ------------------------------------------------------
    #
    # Zero is invalid because it would erase the signal.
    # Negative multipliers are permitted: some products
    # require sign inversion during scaling.

    if feature["multiplier"] == 0:
        raise ValueError(
            "'multiplier' must not be zero."
        )

    # ------------------------------------------------------
    # Reducer output name
    # ------------------------------------------------------

    if feature["reducer_output"] not in VALID_REDUCER_OUTPUTS:
        raise ValueError(
            f"'reducer_output' must be one of "
            f"{sorted(VALID_REDUCER_OUTPUTS)}, got "
            f"'{feature['reducer_output']}'."
        )

    # ------------------------------------------------------
    # Tile scale
    # ------------------------------------------------------
    #
    # Earth Engine accepts tileScale between 1 and 16.

    if "tile_scale" in feature:

        if not 1 <= feature["tile_scale"] <= 16:
            raise ValueError(
                "'tile_scale' must be between 1 and 16."
            )