"""
====================================================================
02_load_kenya_fsc.py

Purpose:
    Load the Kenya FSC reporting units from Google Earth Engine
    and verify the upload.

Author:
    Robert Odingo MSc Thesis
====================================================================
"""

import ee

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

PROJECT_ID = "thesis-project-502010"

ASSET_ID = (
    "projects/thesis-project-502010/assets/thesis/kenya/kenya_fsc_2016"
)

# ------------------------------------------------------------------

print("=" * 70)
print("LOAD KENYA FSC REPORTING UNITS")
print("=" * 70)

ee.Initialize(project=PROJECT_ID)

fc = ee.FeatureCollection(ASSET_ID)

print("\nCounting reporting units...")

count = fc.size().getInfo()

print(f"Reporting units: {count}")

print("\nProperty names:")

properties = fc.first().propertyNames().getInfo()

for p in properties:
    print(f" - {p}")

print("\nFirst reporting unit:")

first = fc.first().toDictionary().getInfo()

for k, v in first.items():
    print(f"{k}: {v}")

print("\n" + "=" * 70)
print("KENYA FSC LOADED SUCCESSFULLY")
print("=" * 70)