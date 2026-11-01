"""
============================================================

countries.py

Country configurations for the MSc Thesis. Single source of
country-specific truth so the feature runners are parameterised by
country, never hardcoded (prevents a stray 'Kenya' constant surviving
into an Ethiopia/Somalia run - a silent-corruption class).

Each country defines:
  name            : display name
  code            : 2-letter short code (KE/ET/SO)
  iso3            : 3-letter ISO code (KEN/ETH/SOM) - WorldPop filter
  asset           : Earth Engine FSC polygon asset
  fsc_polygons    : count of FSC geometry polygons (extraction grid)
  labelled_units  : count of units WITH labels (modelling grid) - this
                    is what the assembler/model panel assert against.
                    Differs from fsc_polygons where a label-less polygon
                    is excluded (Kenya Mvita, Somalia Zeylac; D13).

The fsc_polygons vs labelled_units distinction is explicit because
conflating them is an off-by-one that differs per country:
  Kenya    640 FSC -> 639 labelled  (Mvita dropped)
  Somalia  199 FSC -> 198 labelled  (Zeylac dropped)
  Ethiopia 1040 FSC -> 1040 labelled (no drop)

Author:
Robert Odingo MSc Thesis

============================================================
"""

from config.config import (
    KENYA_FSC,
    ETHIOPIA_FSC,
    SOMALIA_FSC,
)


KENYA = {
    "name": "Kenya",
    "code": "KE",
    "iso3": "KEN",
    "asset": KENYA_FSC,
    "fsc_polygons": 640,
    "labelled_units": 639,      # Mvita (label-less) excluded, D13
}

ETHIOPIA = {
    "name": "Ethiopia",
    "code": "ET",
    "iso3": "ETH",
    "asset": ETHIOPIA_FSC,
    "fsc_polygons": 1040,
    "labelled_units": 1040,     # no label-less polygon
}

SOMALIA = {
    "name": "Somalia",
    "code": "SO",
    "iso3": "SOM",
    "asset": SOMALIA_FSC,
    "fsc_polygons": 199,
    "labelled_units": 198,      # Zeylac (label-less) excluded, D13
}


# Registry for parameterised lookup by name or code.
COUNTRIES = {c["name"].lower(): c for c in (KENYA, ETHIOPIA, SOMALIA)}
COUNTRIES.update({c["code"].lower(): c for c in (KENYA, ETHIOPIA, SOMALIA)})


def get_country(key):
    """
    Resolve a country config by name ('kenya'), code ('ke'), or the
    dict itself. Raises on unknown key so a typo fails loud rather than
    silently defaulting to the wrong country.
    """
    if isinstance(key, dict):
        return key
    k = str(key).strip().lower()
    if k not in COUNTRIES:
        raise KeyError(
            f"Unknown country '{key}'. Valid: "
            f"{sorted(set(c['name'] for c in COUNTRIES.values()))}"
        )
    return COUNTRIES[k]


# Backward-compatible alias: some scripts read config['expected_units'].
# Point it at labelled_units (the modelling grid) explicitly, so the
# old key resolves to the correct (labelled, not FSC) count.
for _c in (KENYA, ETHIOPIA, SOMALIA):
    _c["expected_units"] = _c["labelled_units"]
    _c["reporting_units"] = _c["fsc_polygons"]   # legacy alias -> FSC count
