"""Build the three new Ch5 figures from charter-listed values only.
Before plotting, every value is cross-checked against the Ch5 v8 table text; any mismatch aborts.
Usage: python build_figures.py <ch5_v8.txt> <outdir>
"""
import sys, re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
W = ["2021", "2022", "2023"]

# ---------------- values exactly as listed in the charter ----------------
PREV = {"Kenya": ["21.57", "45.28", "45.36"], "Ethiopia": ["20.78", "36.81", "28.21"],
        "Somalia": ["27.90", "68.41", "63.58"], "Pooled": ["21.96", "42.34", "37.08"]}
BENCH = {"XGBoost (B)": ["0.683", "0.924", "0.881"], "LightGBM (B)": ["0.674", "0.914", "0.854"],
         "Persistence": ["0.476", "0.857", "0.737"], "FEWS NET projection": ["0.514", "0.869", "0.835"]}
BC_IN = {"XGBoost": ["+0.117", "+0.036", "+0.021"], "LightGBM": ["+0.100", "+0.030", "+0.020"]}
BC_TR = {"XGBoost": ["+0.002", "+0.040", "+0.039"], "LightGBM": ["−0.005", "+0.034", "+0.036"]}

# ---------------- cross-check against the chapter's own tables ----------------
text = open(src, encoding="utf8").read()
def block(caption_start):
    i = text.index(caption_start)
    blk = text[i:].split("\n\n", 1)[0].split("\n")[1:]
    return [l.split("\t") for l in blk if "\t" in l]

t51 = {r[0]: r[1:4] for r in block("Table 5.1 —")[1:]}
rows55 = block("Table 5.5 —")[1:]
fews_i = next(i for i, r in enumerate(rows55) if r[0] == "FEWS-aligned rows")
fews = rows55[fews_i:fews_i + 3]
t55 = {k: [r[2 + j] for r in fews] for j, k in enumerate(BENCH)}
assert [r[1] for r in fews] == W
t58, model = {}, None
for r in block("Table 5.8 —")[1:]:
    model = r[0] or model
    if r[1] == "B − C": t58[model] = r[2:5]
t514, model = {}, None
for r in block("Table 5.14 —")[1:]:
    model = r[0] or model
    t514.setdefault(model, []).append((r[1], r[5], r[6]))

checks = []
for k in PREV: checks.append((f"T5.1 {k}", PREV[k], t51[k]))
for k in BENCH: checks.append((f"T5.5 FEWS-aligned {k}", BENCH[k], t55[k]))
for k in BC_IN: checks.append((f"T5.8 B−C {k}", BC_IN[k], t58[k]))
for k in BC_TR:
    checks.append((f"T5.14 B−C transferred {k}", BC_TR[k], [x[1] for x in t514[k]]))
    checks.append((f"T5.14 in-pool col {k} = T5.8", BC_IN[k], [x[2] for x in t514[k]]))
bad = [c for c in checks if c[1] != c[2]]
for name, a, b in checks:
    print(f"{'OK ' if a == b else 'MISMATCH'}  {name}: charter {a} | Ch5 v8 {b}")
if bad:
    sys.exit("ABORT: charter values disagree with Ch5 v8 — do not plot.")

num = lambda s: float(s.replace("−", "-").replace("+", ""))

# ---------------- house style (matches Figs 5.1 / 5.2) ----------------
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, AXIS = "#222222", "#4d4d4d", "#e3e3df", "#b9b9b1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": AXIS,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 1.0,
    "pdf.fonttype": 42, "savefig.dpi": 300})

def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.9); ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)

def label(ax, x, v, txt, size=8.5, pad=0.012, span=1.0):
    off = pad * span
    ax.text(x, v + off if v >= 0 else v - off, txt, ha="center",
            va="bottom" if v >= 0 else "top", fontsize=size, color=MUTED, zorder=6,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

def save(fig, name):
    fig.savefig(f"{out}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out}/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---- Figure A: prevalence by country and window (Table 5.1) ----
fig, ax = plt.subplots(figsize=(7.2, 4.2)); style(ax)
cols = {"Kenya": BLUE, "Ethiopia": ORANGE, "Somalia": "#8e5ea2"}
bw, gap = 0.24, 0.02
for j, (c, colr) in enumerate(cols.items()):
    for i in range(3):
        x = i + (j - 1) * (bw + gap); v = num(PREV[c][i])
        ax.bar(x, v, bw, color=colr, edgecolor="white", lw=1.5)
        label(ax, x, v, PREV[c][i], span=80, pad=0.01)
for i in range(3):
    v = num(PREV["Pooled"][i]); half = 1.5 * bw + gap
    ax.plot([i - half, i + half], [v, v], color=INK, lw=1.6, ls=(0, (4, 2)), zorder=5)
    ax.text(i + half + 0.03, v, PREV["Pooled"][i], va="center", ha="left", fontsize=8.5, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(W); ax.set_xlim(-0.55, 2.75); ax.set_ylim(0, 80)
ax.set_xlabel("Validation window"); ax.set_ylabel("Phase 3+ prevalence (%)")
handles = [Patch(color=v, label=k) for k, v in cols.items()] + \
          [Line2D([0], [0], color=INK, lw=1.6, ls=(0, (4, 2)), label="Pooled")]
ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=9.5, ncol=4,
          labelcolor=MUTED, handlelength=1.6, columnspacing=1.2)
save(fig, "Figure_new_A_prevalence_by_country")

# ---- Figure B: benchmark PR-AUC at h = 3, FEWS-aligned rows (Table 5.5) ----
fig, ax = plt.subplots(figsize=(8.6, 4.2)); style(ax)
spec = {"XGBoost (B)": dict(color=BLUE), "LightGBM (B)": dict(color=ORANGE),
        "Persistence": dict(color="#c9c9c4", hatch="////", hatchcolor="#8a8a84"), "FEWS NET projection": dict(color="#6f6f6a")}
legend_names = {"XGBoost (B)": "XGBoost (B)", "LightGBM (B)": "LightGBM (B)",
                "Persistence": "Persistence", "FEWS NET projection": "FEWS NET projection"}
bw, gap = 0.19, 0.015
for j, (k, kw) in enumerate(spec.items()):
    for i in range(3):
        x = i + (j - 1.5) * (bw + gap); v = num(BENCH[k][i])
        ax.bar(x, v, bw, edgecolor="white", lw=1.5, **kw)
        label(ax, x, v, BENCH[k][i], size=8, span=1.0, pad=0.012)
ax.set_xticks(range(3)); ax.set_xticklabels(W); ax.set_ylim(0, 1.0)
ax.set_xlabel("Validation window (h = 3)"); ax.set_ylabel("PR-AUC (FEWS-aligned rows)")
ax.legend(handles=[Patch(facecolor=kw["color"], hatch=kw.get("hatch"), hatchcolor=kw.get("hatchcolor", "white"), edgecolor="white", label=legend_names[k])
                   for k, kw in spec.items()],
          frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4, fontsize=9.5,
          labelcolor=MUTED, handlelength=1.6)
save(fig, "Figure_new_B_benchmark_h3_fews_aligned")

# ---- Figure C: B − C in-pool vs transferred to Somalia (Tables 5.8 + 5.14) ----
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
tint = {"XGBoost": (BLUE, "#bcd5f3"), "LightGBM": (ORANGE, "#f8cdb9")}
bw, gap = 0.36, 0.03
plt.rcParams["hatch.linewidth"] = 1.0
for ax, (p, m) in zip(axes, [("a", "XGBoost"), ("b", "LightGBM")]):
    style(ax); dark, light = tint[m]
    for i in range(3):
        for j, (vals, kw) in enumerate([(BC_IN[m], dict(color=dark)),
                                        (BC_TR[m], dict(facecolor=light, hatch="////", hatchcolor=dark))]):
            x = i + (j - 0.5) * (bw + gap); v = num(vals[i])
            ax.bar(x, v, bw, edgecolor="white", lw=1.5, **kw)
            label(ax, x, v, vals[i], size=8.5, span=0.16, pad=0.02)
    ax.axhline(0, color="#8a8a84", lw=1.0)
    ax.set_xticks(range(3)); ax.set_xticklabels(W); ax.set_ylim(-0.02, 0.14)
    ax.set_title(f"({p}) {m}", loc="left", fontsize=12, color=INK)
    ax.set_xlabel("Validation window (h = 3)")
axes[0].set_ylabel("PR-AUC increment (B − C)")
for ax, m in zip(axes, ["XGBoost", "LightGBM"]):
    dark, light = tint[m]
    ax.legend(handles=[Patch(color=dark, label="In-pool"),
                       Patch(facecolor=light, hatch="////", hatchcolor=dark, edgecolor="white", lw=0, label="Transferred to Somalia")],
              frameon=False, loc="upper right", fontsize=9, labelcolor=MUTED, handlelength=1.6)
fig.tight_layout(w_pad=2.0)
save(fig, "Figure_new_C_increment_inpool_vs_transfer")
print("built 3 figures ->", out)
