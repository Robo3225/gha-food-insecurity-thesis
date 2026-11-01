"""
======================================================================

model_tft.py  (M10.3 — Temporal Fusion Transformer, C1 second model)

C1's deep-learning benchmark, run at Config B (biophysical + phase
history) for the D16 gate. The smoke test (M7.5) is QUARANTINED — its
code and its PR-AUC 1.000 do not inform this build; constructed fresh
against the frozen v2 panels through the M10.1 harness.

Settled parameters:
  - Same explicit lags as XGBoost (D19 fairness: measure architecture,
    not feature access) PLUS the TFT encoder (its architectural extra).
  - Encoder length FIXED at 6 months (not tuned): explicit 9/12-mo lags
    already carry long-range signal; encoder does short-range structure.
  - Target: ORDINAL phase (D9) — predict fews_ipc(1-5) at t+h, threshold
    predicted median >=3 for binary eval. Preserves quantile outputs
    (P10/P50/P90) for C2 uncertainty.
  - Horizon-shifted target (D23), D25-canonical rows, SHARED binary eval
    (same function XGBoost used) -> like-for-like C1 comparison.

D16 gate (evidence-driven):
  1. Stability: full-scale training without collapse.
  2. Discrimination: TFT-B within 0.05 PR-AUC of XGBoost-B (h=3).
  3. Calibration: 70-90% coverage on the nominal 80% interval (P10-P90).
Pass -> full four-arm TFT. Fail -> LightGBM quantile-regression kill-
switch (D16).

Runs in gee_env (pytorch-forecasting + MPS). Not in this sandbox.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness_data import (
    load_all, CANONICAL_PREDICTORS, TARGET_ORDINAL, UNIT_COL,
)
from harness_target import add_horizon_targets, assert_target_not_leaking
from harness_splits import all_splits, TEST_WINDOWS
from harness_eval import evaluate_binary, evaluate_baselines
from harness_fews import build_fews_baseline_rows, fews_baseline_for_split
from harness_phase_history import (
    add_phase_history, assert_phase_history_safe, PHASE_HISTORY_COLS,
)

ENCODER_LENGTH = 6          # FIXED — not tuned
PHASE3_THRESHOLD = 3
QUANTILES = [0.1, 0.5, 0.9]  # P10/P50/P90 for uncertainty + calibration
MAX_EPOCHS = 30
RANDOM_SEED = 42

# Config B predictor set (same as XGBoost Config B): biophysical + phase history
CONFIG_B_REALS = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def _build_tsds(full_df, horizon, predictors, training_cutoff):
    """
    Construct the TimeSeriesDataSet on the FULL contiguous per-unit series
    (no row-dropping — dropping rows creates encoder gaps). The walk-forward
    boundary is a TIME CUTOFF, not row removal.

    Target: the CONTEMPORANEOUS ordinal phase (fews_ipc at t). The library's
    native horizon (max_prediction_length = horizon) reaches t+h, so the
    h-step-ahead prediction IS the t+h forecast. This is the per-architecture
    construction of the SAME task XGBoost does via the explicit D23-shifted
    target — same target period, scored identically against y_binary_h{h} on
    D25 rows through the shared evaluation (D19 principle: each architecture
    reaches the horizon its natural way; the comparison measures architecture).

    Leakage: encoder = 6 prior observed steps [t-5..t]; decoder = [t+1..t+h],
    strictly future of the encoder. training_cutoff limits TRAINING decoder
    windows to those ending before the test window (D23 chronology as a time
    filter). Test prediction uses from_dataset with the test window.

    NaN handling: the contemporaneous ordinal target has NaN only at genuine
    coverage gaps (D6); allow_missing_timesteps stays False because the panel
    is contiguous in TIME (every month present per unit) — the target may be
    NaN at a step but the step exists. We forward-fill the target for encoder
    continuity (operational last-known phase, D10) and rely on the observed
    binary truth for SCORING, not the filled target.
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder

    df = full_df.copy()
    df["unit"] = df[UNIT_COL].astype(str)
    df["climate_class"] = df["climate_class"].astype(str)
    df["country"] = df["country"].astype(str)
    # contemporaneous ordinal phase as target; forward-fill within unit for
    # encoder continuity (last-known phase, D10), so no NaN breaks the build.
    df = df.sort_values(["unit", "t_index"])
    df["target_ord"] = (
        df.groupby("unit")[TARGET_ORDINAL].ffill().astype(float)
    )

    # Leakage-safe within-unit forward-fill of PREDICTOR NaNs (D11 unfillable
    # NDVI cell, LST cloudy-null lags). TFT cannot hold NaN in predictor
    # tensors; XGBoost handled them natively. Forward-fill = last-observed
    # carried forward (backward-looking, within-unit -> no future info). A
    # fill-indicator flags where a value was imputed (good practice).
    for col in predictors:
        if col in df.columns and df[col].isna().any():
            df[f"{col}_was_filled"] = df[col].isna().astype(float)
            df[col] = df.groupby("unit")[col].ffill()
            # any residual leading NaN (no prior obs in unit) -> back-fill is
            # NOT allowed (would pull future); fill with 0 (neutral) + flag set.
            df[col] = df[col].fillna(0.0)

    # drop leading rows where the target itself was never observed (contiguity
    # preserved from first observed month onward)
    df = df[df["target_ord"].notna()].reset_index(drop=True)

    # Keep the TFT feature access exactly aligned with the declared Config-B
    # set. The *_was_filled columns are used internally for leakage-safe
    # imputation/auditing, but are NOT model predictors. target_ord is the
    # target and must NOT also be supplied as an input real. This keeps the
    # TFT input dimensionality identical to Config B and removes the
    # 27-vs-25 architecture mismatch.
    # climate_class / country are CATEGORICALS -> static_categoricals ONLY.
    # They must NOT also be in the reals list (diagnose_tft_dims proved
    # climate_class was leaking into reals). Exclude them here.
    _CATEGORICALS = ("climate_class", "country")
    tft_reals = [c for c in predictors if c not in _CATEGORICALS]

    tsds = TimeSeriesDataSet(
        df[df["t_index"] <= training_cutoff],
        time_idx="t_index",
        target="target_ord",
        group_ids=["unit"],
        max_encoder_length=ENCODER_LENGTH,
        max_prediction_length=horizon,     # NATIVE h-step horizon -> t+h
        static_categoricals=["climate_class", "country"],
        time_varying_unknown_reals=tft_reals,
        categorical_encoders={
            "climate_class": NaNLabelEncoder(add_nan=True),
            "country": NaNLabelEncoder(add_nan=True),
        },
        target_normalizer=GroupNormalizer(groups=["unit"]),
        allow_missing_timesteps=False,     # panel contiguous in time per unit
        add_relative_time_idx=True,
        add_target_scales=True,
    )
    return tsds, df


def train_tft_one_split(df, split):
    """
    Train a TFT on one walk-forward split's train rows, predict on test.
    Returns median (P50) and P10/P90 for the test rows, plus training
    stability info. Ordinal prediction; binary = (P50 >= 3).
    """
    import torch
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping

    pl.seed_everything(RANDOM_SEED, workers=True)

    h = split["horizon"]
    t_lo = split["t_lo"]   # test window start (predicted-period)

    # Training cutoff: training decoder windows must END before the test
    # window starts. A decoder window ending at t+h; require t+h < t_lo, i.e.
    # the latest training decoder-end is t_lo - 1. Give the dataset the full
    # series up to t_lo-1 as the training-eligible region (encoder lookback
    # is included automatically from earlier steps).
    training_cutoff = t_lo - 1

    tsds, prepared_df = _build_tsds(df, h, CONFIG_B_REALS, training_cutoff)

    # device: MPS on Apple Silicon
    accel = "mps" if torch.backends.mps.is_available() else "cpu"

    train_loader = tsds.to_dataloader(train=True, batch_size=256, num_workers=0)

    # Architecture diagnostic: report the exact feature inventory used by TFT.
    # This makes any future dimensionality mismatch immediately auditable.
    print(f"  TFT inputs: Config-B={len(CONFIG_B_REALS)}, "
          f"fill_flags_audit_only={len([c for c in prepared_df.columns if c.endswith('_was_filled')])}, "
          f"total_real={len(tsds.reals)}")
    print(f"  TFT reals: {tsds.reals}")

    tft = TemporalFusionTransformer.from_dataset(
        tsds,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.2,
        loss=QuantileLoss(quantiles=QUANTILES),
        log_interval=0,
        optimizer="adam",
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS, accelerator=accel, devices=1,
        gradient_clip_val=0.1, enable_progress_bar=False,
        enable_checkpointing=False, logger=False,
        callbacks=[EarlyStopping(monitor="train_loss", patience=5, mode="min")],
    )

    # --- STABILITY diagnostic: capture whether loss converges / NaNs ---
    stable = True
    try:
        trainer.fit(tft, train_dataloaders=train_loader)
        final_loss = float(trainer.callback_metrics.get("train_loss", np.nan))
        if np.isnan(final_loss):
            stable = False
    except Exception as e:
        print(f"  TFT training FAILED (instability): {e}")
        return {"stable": False, "error": str(e)}

    # --- predict the test window natively (h steps ahead) ---
    # Build a prediction dataset over the full prepared series; the library
    # generates, per unit, the decoder window [encoder_end+1 .. +h]. We want
    # decoder windows whose target period falls in the test window. Use
    # predict=True to get the last available window per unit up to each cutoff;
    # to cover all test periods, decode over the full series and filter to
    # target periods in [t_lo, t_hi]. Returns per-(unit, decoder-step) quantiles.
    pred_tsds = TimeSeriesDataSet.from_dataset(
        tsds, prepared_df, predict=False, stop_randomization=True,
    )
    pred_loader = pred_tsds.to_dataloader(train=False, batch_size=256, num_workers=0)

    raw = tft.predict(pred_loader, mode="quantiles", return_index=True)
    # pytorch-forecasting returns a Prediction namedtuple (output, index) when
    # return_index=True in recent versions; handle both shapes.
    if hasattr(raw, "output"):
        q = np.asarray(raw.output); idx = raw.index
    else:
        q = np.asarray(raw[0]); idx = raw[1]
    # q shape: (n_windows, horizon, n_quantiles); take the FINAL decoder step
    # (the t+h forecast) of each window.
    q = q[:, -1, :] if q.ndim == 3 else q.reshape(q.shape[0], -1)
    p10, p50, p90 = q[:, 0], q[:, 1], q[:, 2]

    # idx gives (unit, time_idx) of each prediction's decoder START; the target
    # period is that + h - 1 (final decoder step). Align to test rows.
    idx = idx.copy()
    idx["target_period"] = idx["t_index"] + (h - 1)
    idx["p10"], idx["p50"], idx["p90"] = p10, p50, p90

    return {"stable": stable, "final_loss": final_loss,
            "pred_index": idx,   # unit, target_period, p10/p50/p90
            "quantiles_ordered": bool(np.all(p10 <= p50) and np.all(p50 <= p90))}



# quantile levels the TFT was trained to emit (QUANTILES = [0.1, 0.5, 0.9])
_Q_LEVELS = np.array([0.10, 0.50, 0.90])


def _prob_phase_ge3(p10, p50, p90, threshold=PHASE3_THRESHOLD):
    """
    Continuous P(phase >= threshold) from the predicted quantiles, per row.
    Builds a piecewise-linear CDF through (p10,0.10),(p50,0.50),(p90,0.90),
    extrapolated flat beyond the ends, and returns 1 - CDF(threshold). This
    is a continuous score in [0,1] preserving ranking, for a like-for-like
    PR-AUC vs XGBoost's continuous probability (not a P50>=3 hard threshold).
    """
    p10 = np.asarray(p10, float); p50 = np.asarray(p50, float); p90 = np.asarray(p90, float)
    n = len(p50)
    out = np.empty(n, float)
    for i in range(n):
        xs = np.array([p10[i], p50[i], p90[i]])
        ys = _Q_LEVELS.copy()
        # enforce monotone x for interpolation (guards q-ordering violations)
        order = np.argsort(xs)
        xs = xs[order]; ys = ys[order]
        # ensure strictly increasing xs for np.interp (jitter ties)
        for j in range(1, len(xs)):
            if xs[j] <= xs[j-1]:
                xs[j] = xs[j-1] + 1e-6
        # CDF at threshold: interp cumulative prob; flat beyond ends
        cdf_at = np.interp(threshold, xs, ys, left=ys[0], right=ys[-1])
        out[i] = 1.0 - cdf_at   # P(phase >= threshold)
    return np.clip(out, 0.0, 1.0)


def run_gate():
    section("M10.3 — TFT Config B, D16 gate (h=3 like-for-like vs XGBoost-B)")
    df = load_all()
    df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)
    df = add_phase_history(df)
    assert_phase_history_safe(df)
    splits = all_splits(df, verbose=False)
    fews_rows = build_fews_baseline_rows(df)

    # gate is decided at h=3 (architectural comparison is meaningful there)
    h3_splits = [s for s in splits if s["horizon"] == 3]

    print(f"\n{'window':<8}{'stable':>8}{'q_ord':>7}{'TFT_PR':>9}"
          f"{'persist':>9}{'FEWS':>8}{'cover80':>9}")
    print("-" * 60)

    results = []
    for s in h3_splits:
        yr = s["test_year"]
        out = train_tft_one_split(df, s)
        if not out.get("stable", False):
            print(f"{yr}/h3   UNSTABLE — training collapsed (kill-switch candidate)")
            results.append({"year": yr, "stable": False})
            continue

        # Align TFT predictions (unit, target_period) to test rows' truth.
        # Build a lookup of test rows keyed on (unit, target absolute period).
        ycol_bin = "y_binary_h3"; ycol_ord = "y_ordinal_h3"
        test_df = df.loc[s["test_idx"]].copy()
        test_df["unit"] = test_df[UNIT_COL].astype(str)
        # target absolute period of a test row = its t_index + h (predicted period)
        test_df["target_period"] = test_df["t_index"] + s["horizon"]
        test_df = test_df[test_df[ycol_bin].notna()]  # D25: scorable rows

        pi = out["pred_index"].copy()
        # pred target_period is in the same t_index space; join on (unit, period)
        merged = test_df.merge(
            pi[["unit", "target_period", "p10", "p50", "p90"]],
            on=["unit", "target_period"], how="inner",
        )
        if len(merged) == 0:
            print(f"{yr}/h3   no aligned TFT predictions — check horizon indexing")
            results.append({"year": yr, "stable": True, "aligned": 0})
            continue

        y_true = merged[ycol_bin].to_numpy()
        # CONTINUOUS P(phase>=3) from the quantile distribution — like-for-like
        # with XGBoost's predict_proba (NOT a P50>=3 hard threshold, which would
        # collapse ranking info and depress PR-AUC unfairly).
        tft_prob = _prob_phase_ge3(
            merged["p10"].to_numpy(), merged["p50"].to_numpy(),
            merged["p90"].to_numpy(),
        )
        tft_pr = evaluate_binary(y_true, tft_prob)["all"]["pr_auc"]

        base = evaluate_baselines(df, s)
        persist_pr = base["persistence"]["all"]["pr_auc"]
        fews = fews_baseline_for_split(df, fews_rows, s, proj_type="near")
        fews_pr = fews["fews"]["all"]["pr_auc"] if fews else np.nan

        # CALIBRATION coverage: true ordinal (at t+h) within [P10,P90] (nominal 80%)
        y_ord = merged[ycol_ord].to_numpy()
        finite = np.isfinite(y_ord)
        cover80 = float(np.mean((y_ord[finite] >= merged["p10"].to_numpy()[finite]) &
                                (y_ord[finite] <= merged["p90"].to_numpy()[finite]))) if finite.any() else np.nan

        print(f"{yr}/h3   {'yes':>8}{str(out['quantiles_ordered']):>7}"
              f"{tft_pr:>9.3f}{persist_pr:>9.3f}"
              f"{(fews_pr if not np.isnan(fews_pr) else float('nan')):>8.3f}"
              f"{cover80:>9.2f}")

        results.append({"year": yr, "stable": True,
                        "quantiles_ordered": out["quantiles_ordered"],
                        "tft_pr_auc": tft_pr, "persistence_pr_auc": persist_pr,
                        "fews_pr_auc": fews_pr, "coverage80": cover80,
                        "final_loss": out["final_loss"]})

    section("D16 GATE VERDICT")
    print("Criteria: (1) stable  (2) TFT-B within 0.05 PR-AUC of XGB-B  (3) 80% cover in [70,90]%")
    print("XGB-B h=3 (for comparison): 2021=0.733  2022=0.945  2023=0.890")
    print("Compare TFT_PR above; report the three criteria per window.")
    print("PASS all -> full four-arm TFT. FAIL -> LightGBM quantile kill-switch (D16).")
    pd.DataFrame(results).to_csv("outputs/diagnostics/m10_tft_gate.csv", index=False)
    print("\nsaved: outputs/diagnostics/m10_tft_gate.csv")
    return results


if __name__ == "__main__":
    run_gate()
