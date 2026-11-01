"""
diagnose_tft_dims.py

D16-support diagnostic: find WHY the TFT sees a different number of reals
than its layer expects (the 27-vs-25 mismatch). Does NOT change the model
or train — it constructs the TimeSeriesDataSet exactly as model_tft does
and prints the full variable inventory, so the extra reals are identified
by measurement, not guess.

Hypotheses to test:
  - climate_class / country (categoricals) leaking into reals because they
    are in CANONICAL_PREDICTORS -> CONFIG_B_REALS -> time_varying_unknown_reals.
  - pytorch-forecasting auto-added reals (relative_time_idx, target scales,
    encoder_length) inflating the count beyond declared features.

Run: python -m scripts.04_diagnostics.diagnose_tft_dims
"""
from __future__ import annotations
import numpy as np, pandas as pd

from harness_data import load_all, CANONICAL_PREDICTORS, TARGET_ORDINAL, UNIT_COL
from harness_target import add_horizon_targets
from harness_phase_history import add_phase_history, PHASE_HISTORY_COLS

ENCODER_LENGTH = 6
CONFIG_B_REALS = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS


def main():
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder

    df = load_all(); df = add_horizon_targets(df); df = add_phase_history(df)
    df["unit"] = df[UNIT_COL].astype(str)
    df["climate_class"] = df["climate_class"].astype(str)
    df["country"] = df["country"].astype(str)
    df = df.sort_values(["unit", "t_index"])
    df["target_ord"] = df.groupby("unit")[TARGET_ORDINAL].ffill().astype(float)
    for col in CONFIG_B_REALS:
        if col in df.columns and df[col].isna().any():
            df[col] = df.groupby("unit")[col].ffill().fillna(0.0)
    df = df[df["target_ord"].notna()].reset_index(drop=True)

    print("\n--- DECLARED ---")
    print(f"CONFIG_B_REALS ({len(CONFIG_B_REALS)}): {CONFIG_B_REALS}")
    print(f"climate_class in CONFIG_B_REALS? {'climate_class' in CONFIG_B_REALS}")
    print(f"country in CONFIG_B_REALS? {'country' in CONFIG_B_REALS}")

    # exclude categoricals from the reals we pass (the likely fix)
    tft_reals = [c for c in CONFIG_B_REALS if c not in ("climate_class", "country")]
    print(f"\ntft_reals after excluding categoricals ({len(tft_reals)}): {tft_reals}")

    # build EXACTLY as model_tft does (with the categoricals still in reals -
    # the current buggy state) to see the inflated count
    print("\n--- CURRENT (buggy) BUILD: climate_class left in reals ---")
    tsds_buggy = TimeSeriesDataSet(
        df[df["t_index"] <= 40],
        time_idx="t_index", target="target_ord", group_ids=["unit"],
        max_encoder_length=ENCODER_LENGTH, max_prediction_length=3,
        static_categoricals=["climate_class", "country"],
        time_varying_unknown_reals=list(CONFIG_B_REALS),   # includes climate_class!
        categorical_encoders={"climate_class": NaNLabelEncoder(add_nan=True),
                              "country": NaNLabelEncoder(add_nan=True)},
        target_normalizer=GroupNormalizer(groups=["unit"]),
        allow_missing_timesteps=False, add_relative_time_idx=True,
        add_target_scales=True,
    )
    print(f"tsds.reals ({len(tsds_buggy.reals)}): {tsds_buggy.reals}")
    print(f"static_categoricals: {tsds_buggy.static_categoricals}")
    print(f"time_varying_unknown_reals: {tsds_buggy.time_varying_unknown_reals}")

    print("\n--- FIXED BUILD: categoricals excluded from reals ---")
    tsds_fixed = TimeSeriesDataSet(
        df[df["t_index"] <= 40],
        time_idx="t_index", target="target_ord", group_ids=["unit"],
        max_encoder_length=ENCODER_LENGTH, max_prediction_length=3,
        static_categoricals=["climate_class", "country"],
        time_varying_unknown_reals=tft_reals,              # categoricals excluded
        categorical_encoders={"climate_class": NaNLabelEncoder(add_nan=True),
                              "country": NaNLabelEncoder(add_nan=True)},
        target_normalizer=GroupNormalizer(groups=["unit"]),
        allow_missing_timesteps=False, add_relative_time_idx=True,
        add_target_scales=True,
    )
    print(f"tsds.reals ({len(tsds_fixed.reals)}): {tsds_fixed.reals}")

    print("\n--- VERDICT ---")
    print(f"buggy reals count: {len(tsds_buggy.reals)}")
    print(f"fixed reals count: {len(tsds_fixed.reals)}")
    print("If buggy=27 and the difference is climate_class (+ any auto-added),")
    print("the fix is: exclude climate_class/country from time_varying_unknown_reals")
    print("(they belong ONLY in static_categoricals). Re-check the model builds.")


if __name__ == "__main__":
    main()
