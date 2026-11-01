import rasterio

# Path to the Köppen-Geiger source raster (Beck et al. 2023, present-day
# 1991-2020, 1 km). Obtain it from https://www.gloh2o.org/koppen/ and set
# KOPPEN_TIF, or pass the path as the first argument.
import os, sys
path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "KOPPEN_TIF", "koppen_geiger_0p00833333.tif")

with rasterio.open(path) as src:
    print("=" * 60)
    print("KÖPPEN-GEIGER TIFF INSPECTION")
    print("=" * 60)
    print("CRS              :", src.crs)
    print("Width            :", src.width)
    print("Height           :", src.height)
    print("Bands            :", src.count)
    print("Resolution       :", src.res)
    print("Data type        :", src.dtypes[0])
    print("NoData           :", src.nodata)
    print("Bounds           :", src.bounds)
    print("Compression      :", src.compression)
    print("Color interpreter:", src.colorinterp)