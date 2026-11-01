"""
check_taxonomy.py

D21 §648 cross-country taxonomy check: confirm the Köppen climate-class
categorical has SHARED levels across all three countries' frozen static
panels, so the feature vector's categorical is consistent (C3 feature-
identity requirement).

Runs assert_shared_taxonomy on the per-country unit-level modal sets:
  - validates every class is a real Köppen code (1-30)
  - requires a non-empty intersection (shared levels, not disjoint schemes)

This runs once, at end of M9, when all three static panels exist.

Usage: python -m scripts.04_diagnostics.check_taxonomy
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from scripts.gee.exports.build_static import (
    collect_climate_classes,
    assert_shared_taxonomy,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

COUNTRIES = ["kenya", "ethiopia", "somalia"]


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    section("CROSS-COUNTRY KÖPPEN TAXONOMY CHECK (D21 §648)")

    class_sets = {}
    for slug in COUNTRIES:
        path = PROCESSED / f"{slug}_static_panel.csv"
        if not path.exists():
            raise RuntimeError(f"Static panel missing: {path}")
        df = pd.read_csv(path)
        class_sets[slug.capitalize()] = collect_climate_classes(df)

    # assert_shared_taxonomy prints per-country sets, union, intersection,
    # and raises on invalid codes or empty intersection.
    assert_shared_taxonomy(class_sets)

    # Explicit intersection report for the record.
    inter = set.intersection(*class_sets.values())
    union = set().union(*class_sets.values())
    section("SUMMARY")
    print(f"Shared across all three (intersection): {sorted(inter)}")
    print(f"Full code space (union)               : {sorted(union)}")
    print(f"Per-country counts: "
          + ", ".join(f"{c}={len(s)}" for c, s in class_sets.items()))
    print("\n\u2713 Categorical levels are shared; feature-vector identity "
          "holds on climate_class across all three countries.")


if __name__ == "__main__":
    main()
