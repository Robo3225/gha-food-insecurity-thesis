import rasterio
import numpy as np

# Path to the Köppen-Geiger source raster (Beck et al. 2023, present-day
# 1991-2020, 1 km). Obtain it from https://www.gloh2o.org/koppen/ and set
# KOPPEN_TIF, or pass the path as the first argument.
import os, sys
path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "KOPPEN_TIF", "koppen_geiger_0p00833333.tif")

with rasterio.open(path) as src:

    print("=" * 70)
    print("KÖPPEN-GEIGER CLASS ENCODING INSPECTION")
    print("=" * 70)

    print("File:", path)
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Shape:", src.height, "rows x", src.width, "columns")
    print("Data type:", src.dtypes[0])
    print("NoData:", src.nodata)

    # Read the raster
    data = src.read(1)

    # Unique values and their pixel counts
    values, counts = np.unique(data, return_counts=True)

    print("\nUnique raster values:")
    print("-" * 70)

    for value, count in zip(values, counts):
        percentage = (count / data.size) * 100
        print(f"Value {value:3d} : {count:12,} pixels ({percentage:6.2f}%)")

    # Palette / color table
    print("\nColor table:")
    print("-" * 70)

    colormap = src.colormap(1)

    if colormap:
        for value, color in colormap.items():
            print(f"Value {value:3d} : RGB{color[:3]}")

    else:
        print("No color table found.")

    print("\n" + "=" * 70)
    print("END")
    print("=" * 70)