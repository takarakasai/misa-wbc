#!/usr/bin/env python3
"""Go2 leg singularity study: degraded-tick heatmap, undamped vs damped,
mirroring analyze_singularity.py's structure for the Panda study."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--undamped-dir", required=True)
ap.add_argument("--damped-dir", required=True)
ap.add_argument("--out-dir", required=True)
args = ap.parse_args()

UNDAMPED_DIR = Path(args.undamped_dir)
DAMPED_DIR = Path(args.damped_dir)
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(exist_ok=True, parents=True)

FORMULATIONS = ["explicit", "accel", "force"]
BACKENDS = ["activeset", "ipm", "admm", "clarabel"]
BACKEND_LABEL = {"activeset": "ActiveSet", "ipm": "Ipm", "admm": "Admm", "clarabel": "Clarabel"}
FORM_LABEL = {"explicit": "Explicit", "accel": "AccelSpace", "force": "ForceSpace"}


def load(d, form, backend):
    rows = []
    with open(d / f"sing_{form}_{backend}.csv") as f:
        for line in f:
            rows.append([float(v) for v in line.strip().split(",")])
    return np.array(rows)


def degraded_pct(rows):
    status = rows[:, -2]
    return 100.0 * (status == 1).sum() / len(status)


summary = []
for form in FORMULATIONS:
    for backend in BACKENDS:
        u = load(UNDAMPED_DIR, form, backend)
        d = load(DAMPED_DIR, form, backend)
        summary.append(dict(form=form, backend=backend,
                             undamped_pct=degraded_pct(u), damped_pct=degraded_pct(d)))

with open(OUT_DIR / "go2_singularity_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
labels = [f"{FORM_LABEL[r['form']]}\n{BACKEND_LABEL[r['backend']]}" for r in summary]
x = np.arange(len(summary))
width = 0.36
ax.bar(x - width / 2, [r["undamped_pct"] for r in summary], width, color="#9aa7b8", label="undamped")
ax.bar(x + width / 2, [r["damped_pct"] for r in summary], width, color="#5b6ee8", label="damped")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("% ticks degraded")
ax.set_title("Go2 FR-leg singularity approach (D6 joint-limit CBF engaged): degraded ticks, all 12 combinations", fontsize=11, loc="left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#eeeeee", lw=0.8, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "go2_singularity_degraded.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"{'formulation':11s} {'backend':10s} {'undamped%':>10s} {'damped%':>10s}")
for r in summary:
    print(f"{r['form']:11s} {r['backend']:10s} {r['undamped_pct']:10.1f} {r['damped_pct']:10.1f}")
print("\nWrote:", OUT_DIR / "go2_singularity_degraded.png")
