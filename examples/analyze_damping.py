#!/usr/bin/env python3
"""Compare undamped vs cartesian_acceleration_damped across the same
3x4 Formulation x backend grid: does the DLS-style damping actually
help, and where (if anywhere) is it unsafe to apply blindly?

Usage: generate the 12 undamped traces (Sec.5m) and 12 damped traces
(Sec.5n, panda_singularity_demo <form> <backend> damped) first, then:
    python3 analyze_damping.py --undamped-dir <dir> --damped-dir <dir> --out-dir <dir>
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--undamped-dir", required=True, help="dir of undamped sing_<form>_<backend>.csv traces")
ap.add_argument("--damped-dir", required=True, help="dir of damped sing_<form>_<backend>.csv traces")
ap.add_argument("--out-dir", required=True, help="output dir for chart + summary json")
args = ap.parse_args()

UNDAMPED_DIR = Path(args.undamped_dir)
DAMPED_DIR = Path(args.damped_dir)
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(exist_ok=True, parents=True)

IDX_EE = 11
IDX_REF = 14
FORMULATIONS = ["explicit", "accel", "force"]
BACKENDS = ["activeset", "ipm", "admm", "clarabel"]
BACKEND_LABEL = {"activeset": "ActiveSet", "ipm": "Ipm", "admm": "Admm", "clarabel": "Clarabel"}
FORM_LABEL = {"explicit": "Explicit", "accel": "AccelSpace", "force": "ForceSpace"}


def load(d, form, backend):
    path = d / f"sing_{form}_{backend}.csv"
    rows = []
    with open(path) as f:
        for line in f:
            rows.append([float(v) for v in line.strip().split(",")])
    return np.array(rows)


def degraded_pct(rows):
    status = rows[:, -2]
    return 100.0 * (status == 1).sum() / len(status)


results = []
for form in FORMULATIONS:
    for backend in BACKENDS:
        u = load(UNDAMPED_DIR, form, backend)
        d = load(DAMPED_DIR, form, backend)
        results.append(dict(
            form=form, backend=backend,
            undamped_pct=degraded_pct(u), damped_pct=degraded_pct(d),
            damped_ticks=len(d),  # 1195 for the Explicit+Clarabel blow-up (truncated)
        ))

with open(OUT_DIR / "damping_summary.json", "w") as f:
    json.dump(results, f, indent=2)

# ---- grouped bar: undamped vs damped degraded %, all 12 combos ----
fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
labels = [f"{FORM_LABEL[r['form']]}\n{BACKEND_LABEL[r['backend']]}" for r in results]
x = np.arange(len(results))
width = 0.36
bars_u = ax.bar(x - width / 2, [r["undamped_pct"] for r in results], width, color="#9aa7b8", label="undamped")
colors_d = ["#c0392b" if r["damped_ticks"] < 1250 else "#2fbf6e" for r in results]
bars_d = ax.bar(x + width / 2, [r["damped_pct"] for r in results], width, color=colors_d, label="damped")

for i, r in enumerate(results):
    if r["damped_ticks"] < 1250:
        ax.text(x[i] + width / 2, r["damped_pct"] + 2, "DIVERGED\n(truncated)", ha="center", fontsize=7,
                color="#c0392b", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("% ticks degraded")
ax.set_title("cartesian_acceleration_damped: effect on solver degradation, all 12 combinations", fontsize=12, loc="left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#eeeeee", lw=0.8, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "damping_before_after.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"{'formulation':11s} {'backend':10s} {'undamped%':>10s} {'damped%':>10s} {'note':>12s}")
for r in results:
    note = "DIVERGED" if r["damped_ticks"] < 1250 else ""
    print(f"{r['form']:11s} {r['backend']:10s} {r['undamped_pct']:10.1f} {r['damped_pct']:10.1f} {note:>12s}")
print("\nWrote:", OUT_DIR / "damping_before_after.png")
