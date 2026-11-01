"""Figure 5.4 — grouped three-share SHAP bars at h = 3, from c2_shap.csv.
Integrity gates (abort on failure, never reconcile):
  1. biophys + phase + static == 1.0 in every cell (|err| < 1e-6)
  2. biophys_share rounded to 3 dp == Table 5.10 (Ch5 source) for every h=3 AND h=1 cell
Usage: python build_figure_d.py <c2_shap.csv> <ch5.txt> <outdir>
"""
import sys, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

csv_path, ch5, out = sys.argv[1:4]
rows = list(csv.DictReader(open(csv_path)))
W, M = ["2021", "2022", "2023"], {"xgb": "XGBoost", "lgb": "LightGBM"}

# --- gate 1: shares sum to one ---
ok = True
for r in rows:
    s = float(r["biophys_share"]) + float(r["phase_share"]) + float(r["static_share"])
    flag = abs(s - 1) < 1e-6; ok &= flag
    print(f"{'OK ' if flag else 'FAIL'} sum h={r['horizon']} {r['year']} {r['model']}: {s:.9f}")

# --- gate 2: biophysical share == Table 5.10 ---
text = open(ch5, encoding="utf8").read()
blk = text[text.index("Table 5.10 —"):].split("\n\n", 1)[0].split("\n")[2:]
t510 = {l.split("\t")[0]: l.split("\t")[1:5] for l in blk if "\t" in l}   # h3 xgb, h3 lgb, h1 xgb, h1 lgb
col = {("3", "xgb"): 0, ("3", "lgb"): 1, ("1", "xgb"): 2, ("1", "lgb"): 3}
for r in rows:
    mine = f"{float(r['biophys_share']):.3f}"; theirs = t510[r["year"]][col[(r["horizon"], r["model"])]]
    flag = mine == theirs; ok &= flag
    print(f"{'OK ' if flag else 'FAIL'} biophys vs T5.10 h={r['horizon']} {r['year']} {r['model']}: csv {mine} | table {theirs}")
if not ok:
    sys.exit("ABORT: integrity gate failed — do not plot.")

h3 = {(r["model"], r["year"]): r for r in rows if r["horizon"] == "3"}
GROUPS = [("biophys_share", "Biophysical", dict(facecolor="#2e9e6b")),
          ("phase_share", "Phase history", dict(facecolor="#4a4a46")),
          ("static_share", "Static", dict(facecolor="#d6d6d1", hatch="////", hatchcolor="#8a8a84"))]

INK, MUTED, GRID, AXIS = "#222222", "#4d4d4d", "#e3e3df", "#b9b9b1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": AXIS,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
bw, gap = 0.26, 0.02
for ax, (p, m) in zip(axes, [("a", "xgb"), ("b", "lgb")]):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.9); ax.set_axisbelow(True); ax.tick_params(length=0, pad=6)
    for i, y in enumerate(W):
        for j, (key, _, kw) in enumerate(GROUPS):
            v = float(h3[(m, y)][key]); x = i + (j - 1) * (bw + gap)
            ax.bar(x, v, bw, edgecolor="white", lw=1.5, **kw)
            ax.text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=MUTED, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.set_xticks(range(3)); ax.set_xticklabels(W); ax.set_ylim(0, 0.7)
    ax.set_title(f"({p}) {M[m]}", loc="left", fontsize=12, color=INK)
    ax.set_xlabel("Validation window (h = 3)")
axes[0].set_ylabel("Share of SHAP attribution")
fig.legend(handles=[Patch(label=lab, edgecolor="white", **kw) for _, lab, kw in GROUPS],
           frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, fontsize=9.5, labelcolor=MUTED)
fig.tight_layout(w_pad=2.0)
os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{out}/Figure_5_4_shap_group_shares.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print("built Figure 5.4 ->", out)
