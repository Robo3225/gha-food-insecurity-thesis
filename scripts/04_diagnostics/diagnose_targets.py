"""
======================================================================

diagnose_targets.py

Purpose
-------
Two pre-smoke-test checks in one pass:

1. Explain the 13-vs-14 interpolated-cell discrepancy (confirm the
   missing cell is the label-orphaned Mvita polygon).

2. Characterize the class balance and, crucially, the DISTRIBUTION of
   positives across units and time - the numbers that decide whether
   the smoke test targets LEVEL or TRANSITION, and whether country-level
   training is viable or pooling is required.

Read-only. No data is modified.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
XW = ROOT / "data" / "crosswalks" / "kenya_fnid_admin_code_crosswalk.csv"

panel = pd.read_csv(PROC / "kenya_model_panel.csv")
masked = pd.read_csv(PROC / "kenya_ndvi_masked_cells.csv")
crosswalk = pd.read_csv(XW)


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# =====================================================================
# 1. Mvita / interpolated-cell reconciliation
# =====================================================================

section("1. INTERPOLATED-CELL RECONCILIATION")

dropped = set(masked["FNID"]) - set(crosswalk["FNID"])
print(f"Masked cells (NDVI panel)     : {len(masked)}")
print(f"Interpolated in model panel   : {int(panel['ndvi_interpolated'].sum())}")
print(f"Masked FNIDs absent from XW    : {len(dropped)}")

if dropped:
    print("\nFNIDs interpolated but dropped from the modelling panel")
    print("(label-orphaned polygons, correctly excluded):")
    for f in sorted(dropped):
        rows = masked[masked["FNID"] == f][["FNID", "year", "month"]]
        print(rows.to_string(index=False))
    print("\n=> discrepancy explained: these cells belong to units with "
          "no labels\n   and are absent from the modelling panel by design.")
else:
    print("\n\u26a0 No masked FNID is missing from the crosswalk. The "
          "13-vs-14 gap\n   has another cause - investigate before "
          "proceeding.")


# =====================================================================
# 2. Target class balance and distribution
# =====================================================================

section("2. TARGET DISTRIBUTION")

for target in ["phase3plus", "transition"]:
    s = pd.to_numeric(panel[target], errors="coerce")
    labelled = s.notna().sum()
    pos = int(s.sum())
    print(f"\n{target.upper()}")
    print(f"  labelled unit-months : {labelled:,}")
    print(f"  positives            : {pos:,} ({pos/labelled:.2%})")

    # How are positives distributed ACROSS UNITS?
    per_unit = (
        panel.assign(_p=s)
        .groupby("canonical_admin_code")["_p"]
        .sum()
    )
    units_with_any = (per_unit > 0).sum()
    print(f"  units with >=1 pos   : {units_with_any:,} of "
          f"{panel['canonical_admin_code'].nunique():,} "
          f"({units_with_any/panel['canonical_admin_code'].nunique():.1%})")
    print(f"  units with ZERO pos  : {(per_unit == 0).sum():,}")
    print(f"  max pos in one unit  : {int(per_unit.max())}")


# =====================================================================
# 3. Positives per walk-forward test window
# =====================================================================

section("3. POSITIVES PER WALK-FORWARD WINDOW")

panel["year"] = panel["year_month"].str[:4].astype(int)

print(f"{'window':>8} | {'level pos':>10} | {'transition pos':>14}")
print("-" * 40)
for yr in [2021, 2022, 2023]:
    w = panel[panel["year"] == yr]
    lvl = int(pd.to_numeric(w["phase3plus"], errors="coerce").sum())
    trn = int(pd.to_numeric(w["transition"], errors="coerce").sum())
    print(f"{yr:>8} | {lvl:>10,} | {trn:>14,}")

print("\nInterpretation guide:")
print("  Level positives per window in the hundreds+  -> viable at")
print("  country level. Transition positives < ~100 per window are")
print("  thin for training a TFT and likely need country pooling.")
