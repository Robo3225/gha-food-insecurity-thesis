"""
lst_quality_concentration.py

Within-country test (D20 amendment): does the LST
other-quality fraction associate spatially with aridity / crisis
concentration WITHIN each country - i.e. are the high-other-quality
units the same arid crisis units where SPI, the lag finding, and LST
recovery all converged?

Reported as a within-country spatial ASSOCIATION consistent with the
surface-brightness mechanism - NOT retrieval quality causing anything
about the outcome (register discipline).

For each country:
  - join per-unit other_quality_fraction to region (via crosswalk)
  - join to Köppen class (arid = {3,4,6}) from the static panel
  - join to crisis concentration (Phase 3+ count) from the v2 panel
  - test: is other-quality fraction higher in arid units? in crisis units?

Read-only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from config.countries import get_country

ROOT = Path(__file__).resolve().parents[2]


def one_country(slug):
    q = pd.read_csv(ROOT/"data"/"climatology"/f"{slug}_lst_quality.csv")
    static = pd.read_csv(ROOT/"data"/"processed"/f"{slug}_static_panel.csv")
    cw = pd.read_csv(ROOT/"data"/"crosswalks"/f"{slug}_fnid_admin_code_crosswalk.csv")
    v2 = pd.read_csv(ROOT/"data"/"processed"/"v2"/f"{slug}_feature_panel.csv")

    # join fraction <- climate class (arid = 3,4,6)
    m = q.merge(static[["FNID","climate_class"]], on="FNID", how="inner")
    m["arid"] = m["climate_class"].isin([3,4,6])

    # crisis concentration per unit: total phase3plus, via crosswalk admin_code
    crisis = (v2.groupby("canonical_admin_code")["phase3plus"].sum()
              .rename("p3plus").reset_index())
    cwm = cw[["FNID","admin_code"]].merge(
        crisis, left_on="admin_code", right_on="canonical_admin_code", how="left")
    m = m.merge(cwm[["FNID","p3plus"]], on="FNID", how="left")
    m["crisis_unit"] = m["p3plus"].fillna(0) > 0

    # associations
    arid_frac = m.loc[m.arid, "other_quality_fraction"].mean()
    nonarid_frac = m.loc[~m.arid, "other_quality_fraction"].mean()
    crisis_frac = m.loc[m.crisis_unit, "other_quality_fraction"].mean()
    noncrisis_frac = m.loc[~m.crisis_unit, "other_quality_fraction"].mean()
    # correlation of fraction with crisis intensity (Spearman, robust)
    valid = m.dropna(subset=["other_quality_fraction","p3plus"])
    rho = valid["other_quality_fraction"].corr(valid["p3plus"], method="spearman")

    return {
        "slug": slug,
        "n": len(m),
        "n_arid": int(m.arid.sum()),
        "arid_frac": arid_frac,
        "nonarid_frac": nonarid_frac,
        "arid_minus_nonarid": (arid_frac - nonarid_frac) if pd.notna(nonarid_frac) else np.nan,
        "crisis_frac": crisis_frac,
        "noncrisis_frac": noncrisis_frac,
        "crisis_minus_noncrisis": (crisis_frac - noncrisis_frac) if pd.notna(noncrisis_frac) else np.nan,
        "spearman_frac_vs_p3plus": rho,
    }


def main():
    print("="*74)
    print("WITHIN-COUNTRY: LST other-quality fraction vs aridity & crisis")
    print("(spatial association, surface-brightness mechanism - not causal)")
    print("="*74)
    rows = []
    for slug in ["kenya","ethiopia","somalia"]:
        try:
            rows.append(one_country(slug))
        except Exception as e:
            print(f"{slug}: ERROR {e}")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print()
    for _, r in df.iterrows():
        print(f"--- {r['slug'].upper()} (n={r['n']}, arid units={r['n_arid']}) ---")
        print(f"  other-q fraction: arid={r['arid_frac']:.3f}  "
              f"non-arid={r['nonarid_frac'] if pd.notna(r['nonarid_frac']) else float('nan'):.3f}  "
              f"(arid−nonarid = {r['arid_minus_nonarid']:+.3f})")
        print(f"  other-q fraction: crisis={r['crisis_frac']:.3f}  "
              f"non-crisis={r['noncrisis_frac'] if pd.notna(r['noncrisis_frac']) else float('nan'):.3f}  "
              f"(crisis−noncrisis = {r['crisis_minus_noncrisis']:+.3f})")
        print(f"  Spearman(fraction, Phase3+ count) = {r['spearman_frac_vs_p3plus']:+.3f}")
        print()
    print("Read: positive arid−nonarid AND crisis−noncrisis differences, and")
    print("positive Spearman, = within-country, high-other-quality units ARE the")
    print("arid crisis units. Replicating across all three = arid-north through-line")
    print("on a third independent measure at unit level. Weaker in some country ->")
    print("note it, don't suppress.")


if __name__ == "__main__":
    main()
