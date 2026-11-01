"""
======================================================================

crosswalk_config.py

Purpose
-------
Per-country configuration for FNID <-> admin_code crosswalk
construction. Countries encode their FEWS reporting units
differently, so the parse rule and the join key are NOT universal.

  - Ethiopia : units are ADMIN3 districts.
               admin_name = "ADMIN3, ADMIN2, ADMIN1, Country"
               join key    = (ADMIN1, ADMIN2, ADMIN3)
               VERIFIED: 1040/1040 exact match.

  - Kenya    : units are (admin hierarchy x livelihood zone).
               admin_name has THREE formats (see below).
               join key    = (ADMIN1, ADMIN2, ADMIN3, LZNAME)
               VERIFIED (Feb-2026 snapshot): FSC unique on this
               4-field key; (ADMIN1,ADMIN2,LZNAME) alone is NOT
               unique. ADMIN3 is empty for 355 of 640 polygons.

  - Somalia  : SO_Admin2_LHZ shapefile => same family as Kenya.
               PROVISIONAL until Somalia inspection runs.

Kenya format table (empirically derived, Feb-2026 snapshot)
-----------------------------------------------------------
Format 1 (4 components):
    LZNAME, ADMIN2, ADMIN1, Country
    ADMIN3 is absent -> assigned "" to match FSC's canonical
    representation of missing ADMIN3.
    Count: 15,180 unit-months.

Format 2 (5 components, SPLIT livelihood-zone name):
    LZ-part-1, LZ-part-2, ADMIN2, ADMIN1, Country
    The first TWO fields rejoin (with ", ") to form one LZNAME.
    e.g. "Central Highlands, High Potential Zone, Buuri, Meru, Kenya"
    Count: 4,290 unit-months (21.5% of 5-component).

Format 3 (5 components, ADMIN3 present):
    LZNAME, ADMIN3, ADMIN2, ADMIN1, Country
    e.g. "Eastern Pastoral Zone, Liboi, Dadaab, Garissa, Kenya"
    Count: 15,675 unit-months (78.5% of 5-component).

Formats 2 and 3 are POSITIONALLY IDENTICAL (both 5 fields) and can
only be told apart by testing field content against the FSC
reference vocabularies. See parse_admin_name.classify_five_component.
Classification on the Feb-2026 snapshot: 0 unclassified, 0 ambiguous.
That proof is EMPIRICAL AND DATED - the runtime assertion in the
parser is what protects against the dataset changing.

Author:
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations


COUNTRY_CROSSWALK_CONFIG = {

    "Ethiopia": {

        # Single format, ADMIN3-first.
        "admin_name_formats": {
            4: ["ADMIN3", "ADMIN2", "ADMIN1", "COUNTRY"],
        },

        # No ambiguous formats -> no vocabulary classifier needed.
        "requires_classifier": False,

        "join_key": ["ADMIN1", "ADMIN2", "ADMIN3"],

        "fsc_relative_path": [
            "data", "raw", "ethiopia", "ETHIOPIAFSC2", "ET_Admin3C_2021_2.shp",
        ],
        "fsc_columns": ["FNID", "ADMIN1", "ADMIN2", "ADMIN3"],
        "fsc_rename": {},

        "expected_units": 1040,
    },

    "Kenya": {

        # 4-component is unambiguous. 5-component requires the
        # vocabulary classifier to split Format 2 from Format 3,
        # so it is declared as a SET of candidate layouts rather
        # than a single ordering.
        "admin_name_formats": {
            4: ["LZNAME", "ADMIN2", "ADMIN1", "COUNTRY"],
        },

        "ambiguous_formats": {
            5: {
                # Format 2: first two fields rejoin into LZNAME.
                "split_lz": {
                    "layout": ["LZNAME_A", "LZNAME_B", "ADMIN2", "ADMIN1", "COUNTRY"],
                    "rejoin": {"LZNAME": ["LZNAME_A", "LZNAME_B"]},
                },
                # Format 3: ADMIN3 occupies field 1.
                "admin3": {
                    "layout": ["LZNAME", "ADMIN3", "ADMIN2", "ADMIN1", "COUNTRY"],
                    "rejoin": {},
                },
            },
        },

        "requires_classifier": True,

        # ADMIN3 confirmed necessary: (ADMIN1,ADMIN2,LZNAME) is NOT
        # unique across the 640 FSC polygons.
        "join_key": ["ADMIN1", "ADMIN2", "ADMIN3", "LZNAME"],

        "fsc_relative_path": [
            "data", "raw", "kenya", "KENYAFSC", "KE_Admin3_LHZ_2016.2.shp",
        ],
        # LZNAME is used directly - no rename needed, since FEWS-side
        # fields are parsed into the same canonical label.
        "fsc_columns": ["FNID", "ADMIN1", "ADMIN2", "ADMIN3", "LZNAME"],
        "fsc_rename": {},

        "expected_units": 639,
    },

    "Somalia": {

        # VERIFIED (Feb-2026 snapshot): units are ADMIN2 x livelihood
        # zone. ADMIN3 is empty for all 199 FSC polygons, so there is
        # no split-vs-ADMIN3 ambiguity of the kind Kenya has.
        #
        # Every record is structurally FOUR fields:
        #     LZNAME, ADMIN2, ADMIN1, Country
        # but Somalia LZ names contain commas, e.g.
        #     "Southern Inland Pastoral - Camels, Goats, Sheep, and Cattle"
        # so a naive comma split yields 4, 6, or 7 components. Observed
        # distribution: 4 -> 7,645 | 6 -> 1,705 | 7 -> 1,540. The absence
        # of any 5-component record confirms there is no second layout.
        #
        # Parsed RIGHT-ANCHORED: the trailing fields are fixed and
        # everything before them rejoins into LZNAME. This handles any
        # number of internal commas without enumerating counts.
        "right_anchored_format": {
            "trailing": ["ADMIN2", "ADMIN1", "COUNTRY"],
            "leading": "LZNAME",
        },

        "admin_name_formats": {},
        "requires_classifier": False,

        # ADMIN3 retained in the key for cross-country consistency; it
        # is uniformly "" for Somalia on both sides of the merge.
        "join_key": ["ADMIN1", "ADMIN2", "ADMIN3", "LZNAME"],

        "fsc_relative_path": [
            "data", "raw", "somalia", "SOMALIAFSC", "SO_Admin2_LHZ_2016.2.shp",
        ],
        "fsc_columns": ["FNID", "ADMIN1", "ADMIN2", "ADMIN3", "LZNAME"],
        "fsc_rename": {},

        "expected_units": 198,
    },
}


# ---------------------------------------------------------------------
# Accessor + load-time consistency validation
# ---------------------------------------------------------------------

def get_country_config(country: str) -> dict:
    """
    Return the crosswalk config for a country, raising a clear error
    if it is missing or internally inconsistent. Validation happens at
    config load, not mid-run.
    """

    if country not in COUNTRY_CROSSWALK_CONFIG:
        raise KeyError(
            f"No crosswalk config for '{country}'. "
            f"Available: {sorted(COUNTRY_CROSSWALK_CONFIG)}"
        )

    cfg = COUNTRY_CROSSWALK_CONFIG[country]

    # Unambiguous formats: label count must equal component count.
    for count, labels in cfg.get("admin_name_formats", {}).items():
        if len(labels) != count:
            raise ValueError(
                f"{country}: admin_name_formats[{count}] has "
                f"{len(labels)} labels, expected {count}."
            )

    # Ambiguous formats: every candidate layout must match the count.
    for count, candidates in cfg.get("ambiguous_formats", {}).items():
        for name, spec in candidates.items():
            if len(spec["layout"]) != count:
                raise ValueError(
                    f"{country}: ambiguous_formats[{count}]['{name}'] "
                    f"layout has {len(spec['layout'])} labels, "
                    f"expected {count}."
                )

    # Right-anchored countries declare no positional formats.
    ra = cfg.get("right_anchored_format")

    # Every join-key field must be producible by some format.
    producible = set()
    if ra:
        producible |= set(ra["trailing"]) | {ra["leading"]}
    for labels in cfg.get("admin_name_formats", {}).values():
        producible |= set(labels)
    for candidates in cfg.get("ambiguous_formats", {}).values():
        for spec in candidates.values():
            producible |= set(spec["layout"])
            producible |= set(spec.get("rejoin", {}))

    # ADMIN3 may be supplied as "" by formats that lack it.
    producible.add("ADMIN3")

    missing = set(cfg["join_key"]) - producible
    if missing:
        raise ValueError(
            f"{country}: join_key fields {sorted(missing)} are never "
            f"produced by any declared format. Merge key cannot be built."
        )

    return cfg