"""
validate_chirps_spi_signal.py

Country-parameterised diagnostic for a CHIRPS SPI-3 feature panel.

Confirms the drought signal is present, spatially sensible, and
temporally coherent - not just plausible at the national mean. Runs
the same five evidence blocks for any country:

  1. National yearly rainfall / SPI summary
  2. Drought severity distribution (counts at SPI <= -1 .. -3)
  3. Regional breakdown by NAMED admin1 region (via crosswalk ->
     reporting-units), not a hardcoded FNID prefix - so it works for
     Kenya (livelihood-zone) and Ethiopia/Somalia (district) alike
  4. Representative unit from the driest region (data-driven)
  5. Empirical-fallback distribution

Usage:
  python -m scripts.04_diagnostics.validate_chirps_spi_signal --country ethiopia

Outputs (country-stamped):
  outputs/diagnostics/{country}_spi_year_summary.csv
  outputs/diagnostics/{country}_spi_region_summary.csv
  outputs/diagnostics/{country}_spi_example_unit.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config.countries import get_country

ROOT = Path(__file__).resolve().parents[2]


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def load_region_map(slug):
    """
    Build FNID -> region (ADMIN1) via the crosswalk and reporting-units.
    Returns a DataFrame [FNID, admin_code, region] or None if inputs
    are missing (diagnostic still runs, just skips the regional block).
    """
    cw_path = ROOT / "data" / "crosswalks" / f"{slug}_fnid_admin_code_crosswalk.csv"
    ru_path = ROOT / "data" / "interim" / f"{slug}_fews_reporting_units.csv"
    if not (cw_path.exists() and ru_path.exists()):
        return None
    cw = pd.read_csv(cw_path)[["FNID", "admin_code"]]
    ru = pd.read_csv(ru_path)[["admin_code", "ADMIN1"]].rename(
        columns={"ADMIN1": "region"})
    return cw.merge(ru, on="admin_code", how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    args = ap.parse_args()
    cfg = get_country(args.country)
    slug = cfg["name"].lower()

    panel_path = ROOT / "data" / "processed" / f"{slug}_chirps_panel.csv"
    out = ROOT / "outputs" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)

    section(f"{cfg['name'].upper()} CHIRPS SPI DIAGNOSTIC")
    print(f"panel: {panel_path.name}  rows: {len(panel):,}  "
          f"units: {panel['FNID'].nunique()}")

    # --- 1. national yearly summary ---
    section("1. NATIONAL YEARLY SUMMARY")
    ys = (panel.groupby("year")
          .agg(mean_rainfall=("rainfall", "mean"),
               median_rainfall=("rainfall", "median"),
               mean_spi3=("spi3", "mean"),
               median_spi3=("spi3", "median"))
          .round(3))
    print(ys.to_string())
    ys.to_csv(out / f"{slug}_spi_year_summary.csv")

    # --- 2. drought severity distribution ---
    section("2. DROUGHT SEVERITY (counts at SPI <= threshold)")
    thresholds = [-1, -1.5, -2, -2.5, -3]
    rows = []
    for yr in sorted(panel.year.unique()):
        sub = panel[panel.year == yr]
        row = {"year": yr}
        for t in thresholds:
            row[f"SPI<={t}"] = int((sub["spi3"] <= t).sum())
        rows.append(row)
    sev = pd.DataFrame(rows)
    print(sev.to_string(index=False))
    print("\n-> the decisive block: the drought year should show markedly "
          "more SPI<=-2 unit-months than surrounding years. Note WHICH year "
          "peaks (may differ per country - transmission timing, D19).")

    # --- 3. regional breakdown by NAMED region ---
    section("3. REGIONAL BREAKDOWN (by admin1 region, driest first)")
    region_map = load_region_map(slug)
    if region_map is None:
        print("(crosswalk/reporting-units not found - skipping regional block)")
        drought_year = int(sev.set_index("year")["SPI<=-2"].idxmax())
    else:
        pr = panel.merge(region_map[["FNID", "region"]], on="FNID", how="left")
        # use the empirically driest year (most SPI<=-2) as the focus year
        drought_year = int(sev.set_index("year")["SPI<=-2"].idxmax())
        print(f"(focus year = {drought_year}, the year with most SPI<=-2 cells)\n")
        reg = (pr[pr.year == drought_year]
               .groupby("region")
               .agg(mean_spi3=("spi3", "mean"),
                    mean_rainfall=("rainfall", "mean"),
                    n=("spi3", "size"))
               .sort_values("mean_spi3")
               .round(3))
        print(reg.to_string())
        reg.to_csv(out / f"{slug}_spi_region_summary.csv")
        print("\n-> arid regions should sit at the negative (dry) end; wetter "
              "highland/coastal regions at the positive end. A clear gradient "
              "confirms the drought is spatially real, not a mean artifact.")

    # --- 4. representative unit from the driest region ---
    section("4. REPRESENTATIVE UNIT (driest-region example)")
    if region_map is not None:
        pr = panel.merge(region_map[["FNID", "region"]], on="FNID", how="left")
        driest_region = (pr[pr.year == drought_year]
                         .groupby("region")["spi3"].mean().idxmin())
        cand = pr[pr.region == driest_region]
        driest_unit = cand.groupby("FNID")["spi3"].mean().idxmin()
        print(f"driest region: {driest_region}   unit: {driest_unit}")
        ex = panel[panel.FNID == driest_unit].sort_values(["year", "month"])
        print(ex[["year", "month", "rainfall", "spi3", "spi3_empirical"]]
              .to_string(index=False))
        ex.to_csv(out / f"{slug}_spi_example_unit.csv", index=False)
    else:
        print("(no region map - skipping)")

    # --- 5. empirical fallback ---
    section("5. EMPIRICAL FALLBACK DISTRIBUTION")
    emp = panel[panel["spi3_empirical"]]
    print(f"empirical rows: {len(emp):,} ({len(emp)/len(panel)*100:.1f}%)")
    if len(emp):
        print("\nby year:\n" + emp.groupby("year").size().to_string())
        print("\nby month:\n" + emp.groupby("month").size().to_string())

    section("SUMMARY")
    print(f"Country          : {cfg['name']}")
    print(f"SPI-3 range      : {panel['spi3'].min():.2f} .. {panel['spi3'].max():.2f}")
    print(f"Driest year (SPI<=-2 peak): {drought_year}")
    print(f"Empirical rate   : {len(emp)/len(panel)*100:.1f}%")


if __name__ == "__main__":
    main()
