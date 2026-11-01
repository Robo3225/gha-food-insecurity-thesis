"""
====================================================================
01_initialize_gee.py

Purpose:
    Initialize and verify the Google Earth Engine Python API.

Author:
    Robert Odingo MSc Thesis

====================================================================
"""

import ee

PROJECT_ID = "thesis-project-502010"

# ==================================================================

print("=" * 70)
print("GOOGLE EARTH ENGINE INITIALIZATION")
print("=" * 70)

if PROJECT_ID.strip() == "":
    raise ValueError(
        "\nPlease enter your Google Cloud PROJECT_ID "
        "at the top of this script before running it.\n"
    )

try:
    # Try initializing directly
    ee.Initialize(project=PROJECT_ID)
    print("✓ Earth Engine initialized successfully.")

except Exception as e:

    print("\nEarth Engine is not yet authenticated.")
    print("Launching authentication...\n")

    ee.Authenticate()

    ee.Initialize(project=PROJECT_ID)

    print("✓ Authentication successful.")
    print("✓ Earth Engine initialized successfully.")

print("\nChecking server connection...")

try:

    value = ee.Number(5).multiply(4).getInfo()

    print(f"✓ Server response: {value}")

except Exception as e:

    print("✗ Server test failed.")
    print(e)
    raise

print("\nChecking account access...")

try:

    root = ee.data.getAssetRoots()

    print("✓ Connected to Earth Engine.")
    print(f"Number of asset roots: {len(root)}")

    if len(root):

        print("\nAvailable asset roots:")

        for asset in root:
            print(f"  - {asset['id']}")

except Exception as e:

    print("\nCould not retrieve asset roots.")
    print(e)

print("\n" + "=" * 70)
print("EARTH ENGINE IS READY")
print("=" * 70)