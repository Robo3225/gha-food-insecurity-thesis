import geopandas as gpd

# ======================================================
# CHANGE THIS LINE TO SWITCH BETWEEN ADMIN1 / ADMIN2
# ======================================================

shapefile = "ETHIOPIAFSC/ET_Admin3C_2023_3.shp"


# ======================================================

gdf = gpd.read_file(shapefile)

print("=" * 70)
print(shapefile)
print("=" * 70)

print("\nNumber of polygons:")
print(len(gdf))

print("\nCRS:")
print(gdf.crs)

print("\nColumns:")
for c in gdf.columns:
    print("-", c)

print("\nUnique values")

for c in gdf.columns:

    if c == "geometry":
        continue

    try:
        n = gdf[c].nunique()

        print(f"{c}: {n}")

    except:
        pass

print("\nFirst 20 rows")

cols = [c for c in gdf.columns if c != "geometry"]

print(gdf[cols].head(20))