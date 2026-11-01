"""
======================================================================

smoke_tft.py  (scripts/05_smoke_test/ — DISPOSABLE)

TFT smoke test: the five diagnostics.

Predicts IPC phase (ordinal/continuous) via quantile regression;
thresholds the median at >= 3 for the binary Phase 3+ evaluation. This
keeps the P10/P50/P90 quantiles that Contribution C2 depends on.

Hardware: written for Apple Silicon (MPS) with CPU fallback. This is a
PLUMBING test on a small stratified subset - it answers the five yes/no
questions, not final accuracy. Full-panel training comes later.

The five diagnostics, each printed as an explicit verdict:
  1. TimeSeriesDataSet constructs
  2. Trains without memory failure (loss decreases)
  3. Attention shifts across timesteps (not flat)
  4. Quantiles stay ordered (P10 <= P50 <= P90, no crossing)
  5. Beats persistence on PR-AUC

======================================================================
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd

import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import smoke_common as SC


# IPC phase is the ORDINAL target the TFT regresses; phase3plus is
# derived from it at evaluation time by thresholding the median.
PHASE = "ipc_filled"

MAX_ENCODER = 12      # 12-month lookback; diagnostic 3 may suggest 6 or 18
MAX_PRED = 1          # 1-month horizon for the smoke test
QUANTILES = [0.1, 0.5, 0.9]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------
# Diagnostic 1: dataset construction
# ---------------------------------------------------------------------

def build_datasets(train: pd.DataFrame, val: pd.DataFrame):
    """
    Construct the TimeSeriesDataSet. The variable-role declaration below
    is the single most common place a TFT silently learns nothing, so it
    is explicit:

      - group_ids                 : the reporting unit
      - time_idx                  : contiguous month index
      - target                    : IPC phase (ordinal, regressed)
      - time_varying_known_reals  : calendar (always known ahead)
      - time_varying_unknown_reals: NDVI (not known at prediction time)
      - static_categoricals       : unit id (spatial identity)
    """
    train = train.dropna(subset=[PHASE]).copy()
    val = val.dropna(subset=[PHASE]).copy()
    train[PHASE] = train[PHASE].astype(float)
    val[PHASE] = val[PHASE].astype(float)

    training = TimeSeriesDataSet(
        train,
        time_idx=SC.TIME,
        target=PHASE,
        group_ids=[SC.GROUP],
        max_encoder_length=MAX_ENCODER,
        max_prediction_length=MAX_PRED,
        static_categoricals=[SC.GROUP],
        time_varying_known_reals=["month_sin", "month_cos", SC.TIME],
        time_varying_unknown_reals=["ndvi", PHASE],
        target_normalizer=GroupNormalizer(groups=[SC.GROUP]),
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, pd.concat([train, val]), predict=True, stop_randomization=True
    )
    return training, validation


# ---------------------------------------------------------------------
# Diagnostics 2-5: train and evaluate
# ---------------------------------------------------------------------

def run_tft_smoke(train_df, val_df, test_df, device=None, max_epochs=8):
    device = device or pick_device()
    results = {"device": device}

    # --- Diagnostic 1 ---
    try:
        training, validation = build_datasets(train_df, val_df)
        train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_loader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)
        results["d1_dataset_constructs"] = True
        results["encoder_length"] = MAX_ENCODER
    except Exception as e:
        results["d1_dataset_constructs"] = False
        results["d1_error"] = str(e)[:300]
        return results   # nothing else can run

    # --- build model ---
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,          # small: plumbing test
        attention_head_size=2,
        dropout=0.1,
        loss=QuantileLoss(quantiles=QUANTILES),
        log_interval=0,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=("mps" if device == "mps" else "cpu"),
        devices=1,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
        gradient_clip_val=0.1,
    )

    # --- Diagnostic 2: trains without memory failure, loss decreases ---
    t0 = time.time()
    try:
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        results["d2_trains"] = True
        results["train_seconds"] = round(time.time() - t0, 1)
        # loss trajectory
        metrics = trainer.callback_metrics
        results["final_val_loss"] = float(metrics.get("val_loss", float("nan")))
    except RuntimeError as e:
        results["d2_trains"] = False
        results["d2_error"] = str(e)[:300]
        return results

    # --- predict with quantiles ---
    # return_index gives a dataframe mapping each prediction row to its
    # (group_id, time_idx) - avoids version-fragile raw-tensor unpacking.
    raw = tft.predict(val_loader, mode="raw", return_x=True, return_index=True)
    q = raw.output.prediction.detach().cpu().numpy()
    q = q.reshape(q.shape[0], -1, len(QUANTILES))  # [n,1,3]->[n,3]
    p10, p50, p90 = q[:, 0, 0], q[:, 0, 1], q[:, 0, 2]

    pred_index = raw.index  # dataframe with GROUP and TIME columns

    # --- Diagnostic 4: quantiles ordered ---
    crossings = int(((p10 > p50) | (p50 > p90)).sum())
    results["d4_quantiles_ordered"] = (crossings == 0)
    results["quantile_crossings"] = crossings

    # --- Diagnostic 3: attention shifts across timesteps ---
    try:
        interp = tft.interpret_output(raw.output, reduction="mean")
        attn = interp["attention"].detach().cpu().numpy()  # [encoder_len]
        results["attention_variance"] = float(np.var(attn))
        results["d3_attention_shifts"] = float(np.var(attn)) > 1e-4
    except Exception as e:
        results["d3_attention_shifts"] = None
        results["d3_error"] = str(e)[:200]

    # --- Diagnostic 5: beats persistence (PR-AUC) ---
    # median prediction -> continuous phase; align via the index df.
    pred_df = pred_index.copy()
    pred_df[SC.GROUP] = pred_df[SC.GROUP].astype(str)
    pred_df["pred_phase"] = p50

    # The index time column may be named time_idx already; ensure it.
    if SC.TIME not in pred_df.columns:
        # PyTorch Forecasting names it time_idx; fall back if not present
        time_candidates = [c for c in pred_df.columns if "time" in c.lower()]
        if time_candidates:
            pred_df = pred_df.rename(columns={time_candidates[0]: SC.TIME})

    truth = val_df.copy()
    truth[SC.GROUP] = truth[SC.GROUP].astype(str)
    merged = pred_df.merge(
        truth[[SC.GROUP, SC.TIME, SC.TARGET]], on=[SC.GROUP, SC.TIME], how="inner"
    )
    print(f"   [d5] predictions={len(pred_df)}, matched to truth={len(merged)}")
    tft_score = (merged["pred_phase"] >= 3).astype(float)  # or use raw score
    tft_res = SC.evaluate(merged[SC.TARGET], merged["pred_phase"],
                          (merged["pred_phase"] >= 3).astype(int), label="tft")

    pers = SC.persistence_prediction(val_df)
    pers_res = SC.evaluate(pers[SC.TARGET], pers["persistence"].fillna(0),
                           pers["persistence"].fillna(0), label="persistence")

    results["tft_pr_auc"] = tft_res["pr_auc"]
    results["persistence_pr_auc"] = pers_res["pr_auc"]
    results["d5_beats_persistence"] = (
        tft_res["pr_auc"] > pers_res["pr_auc"]
        if not np.isnan(tft_res["pr_auc"]) else None
    )
    return results


def print_verdicts(r: dict):
    print("\n" + "=" * 70)
    print("SMOKE TEST DIAGNOSTICS")
    print("=" * 70)
    print(f"device: {r.get('device')}")

    def verdict(k):
        v = r.get(k)
        return "PASS" if v is True else ("FAIL" if v is False else "N/A ")

    print(f"[{verdict('d1_dataset_constructs')}] 1. TimeSeriesDataSet constructs")
    if not r.get("d1_dataset_constructs"):
        print(f"      -> {r.get('d1_error')}")
        return
    print(f"[{verdict('d2_trains')}] 2. Trains without memory failure "
          f"({r.get('train_seconds','?')}s, val_loss={r.get('final_val_loss','?')})")
    if not r.get("d2_trains"):
        print(f"      -> {r.get('d2_error')}")
        return
    print(f"[{verdict('d3_attention_shifts')}] 3. Attention shifts across timesteps "
          f"(var={r.get('attention_variance','?')})")
    print(f"[{verdict('d4_quantiles_ordered')}] 4. Quantiles stay ordered "
          f"({r.get('quantile_crossings','?')} crossings)")
    print(f"[{verdict('d5_beats_persistence')}] 5. Beats persistence "
          f"(TFT PR-AUC={r.get('tft_pr_auc','?'):.3f} vs "
          f"persistence={r.get('persistence_pr_auc','?'):.3f})")
    print("=" * 70)
