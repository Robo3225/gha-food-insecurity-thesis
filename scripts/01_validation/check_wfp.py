import pandas as pd

# Replace with your filename
df = pd.read_csv("wfp_food_prices_som.csv")

print("\nColumns:")
print(df.columns.tolist())

# Find the date column
date_col = None
for col in df.columns:
    if "date" in col.lower():
        date_col = col
        break

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    print("\nLatest date:")
    print(df[date_col].max())

# Look for admin2 or district columns
possible = [c for c in df.columns if any(
    x in c.lower() for x in
    ["admin", "district", "market"]
)]

print("\nRelevant columns:")
for c in possible:
    print(c)

# Count distinct markets per month
market_col = None
for c in df.columns:
    if "market" in c.lower():
        market_col = c
        break

if market_col and date_col:
    df["month"] = df[date_col].dt.to_period("M")

    counts = (
        df.groupby("month")[market_col]
          .nunique()
          .reset_index(name="distinct_markets")
    )

    print("\nDistinct markets per month:")
    print(counts)
