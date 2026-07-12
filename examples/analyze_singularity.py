#!/usr/bin/env python3
"""Build the comparison table + charts for the singularity-stability
study (Formulation x QP backend, fixed HqpStrategy=NullSpace) from the
12 panda_singularity_demo.rs CSV traces.

Usage: generate the 12 traces first (see wbc_comparison.md Sec.5m for
the exact sweep), then:
    python3 analyze_singularity.py --sing-dir <dir of sing_<form>_<backend>.csv> --out-dir <dir>
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--sing-dir", required=True, help="dir containing sing_<formulation>_<backend>.csv traces")
ap.add_argument("--out-dir", required=True, help="output dir for charts + summary json")
args = ap.parse_args()

SING_DIR = Path(args.sing_dir)
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(exist_ok=True, parents=True)

IDX_T = 1
IDX_EE = 11
IDX_REF = 14

FORMULATIONS = ["explicit", "accel", "force"]
BACKENDS = ["activeset", "ipm", "admm", "clarabel"]
BACKEND_LABEL = {"activeset": "ActiveSet", "ipm": "Ipm", "admm": "Admm", "clarabel": "Clarabel"}
FORM_LABEL = {"explicit": "Explicit", "accel": "AccelSpace", "force": "ForceSpace"}
# Okabe-Ito colorblind-safe categorical set, fixed per backend across all panels.
COLOR = {"activeset": "#0072B2", "ipm": "#E69F00", "admm": "#009E73", "clarabel": "#D55E00"}


def load(form, backend):
    path = SING_DIR / f"sing_{form}_{backend}.csv"
    rows = []
    with open(path) as f:
        for line in f:
            rows.append([float(v) for v in line.strip().split(",")])
    return np.array(rows)


data = {(f, b): load(f, b) for f in FORMULATIONS for b in BACKENDS}

summary = []
for (form, backend), rows in data.items():
    sigma_min = rows[:, -6]
    tau_norm = rows[:, -5]
    tau_max_abs = rows[:, -4]
    qddot_norm = rows[:, -3]
    status = rows[:, -2]
    err = np.linalg.norm(rows[:, IDX_EE:IDX_EE + 3] - rows[:, IDX_REF:IDX_REF + 3], axis=1)
    n_degraded = int((status == 1).sum())
    summary.append(dict(
        form=form, backend=backend,
        sigma_min_min=float(sigma_min.min()),
        err_max=float(err.max()), err_mean=float(err.mean()),
        tau_max_abs_max=float(tau_max_abs.max()),
        qddot_norm_max=float(qddot_norm.max()),
        n_degraded=n_degraded, n_total=int(len(status)),
    ))

with open(OUT_DIR / "singularity_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ---- Figure: 2x2 comparison panel, Explicit formulation across backends ----
fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor="white")
((ax_sigma, ax_err), (ax_qdd, ax_bar)) = axes

for backend in BACKENDS:
    rows = data[("explicit", backend)]
    t = rows[:, IDX_T]
    color = COLOR[backend]
    label = BACKEND_LABEL[backend]

    sigma_min = rows[:, -6]
    ax_sigma.plot(t, sigma_min, color=color, lw=1.4, label=label)

    err = np.linalg.norm(rows[:, IDX_EE:IDX_EE + 3] - rows[:, IDX_REF:IDX_REF + 3], axis=1)
    ax_err.plot(t, err, color=color, lw=1.2, label=label)

    qddot_norm = rows[:, -3]
    ax_qdd.plot(t, np.clip(qddot_norm, 1e-3, None), color=color, lw=1.2, label=label)

ax_sigma.axhline(0.05, color="#999999", lw=0.8, ls="--")
ax_sigma.text(0.02, 0.052, "near-singular threshold", fontsize=7.5, color="#666666", transform=ax_sigma.get_yaxis_transform())
ax_sigma.set_title("Manipulability -- $\\sigma_{min}(J_{lin})$", fontsize=10.5, loc="left")
ax_sigma.set_ylabel("$\\sigma_{min}$ [m/rad]")
ax_sigma.set_xlabel("t [s]")

ax_err.set_title("EE tracking error", fontsize=10.5, loc="left")
ax_err.set_ylabel("$\\|x_{ee} - x_{ref}\\|$ [m]")
ax_err.set_xlabel("t [s]")

ax_qdd.set_yscale("log")
ax_qdd.set_title("Commanded acceleration $\\|\\ddot q\\|$ (log scale)", fontsize=10.5, loc="left")
ax_qdd.set_ylabel("$\\|\\ddot q\\|$ [rad/s$^2$]")
ax_qdd.set_xlabel("t [s]")

# Grouped bar: degraded-tick counts across the full 3x4 grid.
x = np.arange(len(FORMULATIONS))
width = 0.2
for i, backend in enumerate(BACKENDS):
    counts = [data[(form, backend)][:, -2].sum() for form in FORMULATIONS]
    ax_bar.bar(x + (i - 1.5) * width, counts, width, color=COLOR[backend], label=BACKEND_LABEL[backend])
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([FORM_LABEL[f] for f in FORMULATIONS])
ax_bar.set_ylabel("degraded ticks / 1250")
ax_bar.set_title("Solver degradation count (all 3x4 combinations)", fontsize=10.5, loc="left")

for ax in (ax_sigma, ax_err, ax_qdd, ax_bar):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

handles, labels = ax_sigma.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9.5)
fig.suptitle("Panda singularity-approach stability -- Explicit formulation, 4 QP backends", fontsize=12, y=1.08)
fig.tight_layout(rect=[0, 0, 1, 1.0])
fig.savefig(OUT_DIR / "singularity_explicit_backends.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- Figure: full 3x4 grid heatmap-style table of degraded-tick fraction ----
fig2, ax2 = plt.subplots(figsize=(7.2, 3.6), facecolor="white")
grid = np.array([[data[(f, b)][:, -2].sum() / 1250.0 * 100 for b in BACKENDS] for f in FORMULATIONS])
im = ax2.imshow(grid, cmap="Reds", vmin=0, vmax=grid.max())
ax2.set_xticks(range(len(BACKENDS)))
ax2.set_xticklabels([BACKEND_LABEL[b] for b in BACKENDS])
ax2.set_yticks(range(len(FORMULATIONS)))
ax2.set_yticklabels([FORM_LABEL[f] for f in FORMULATIONS])
for i in range(len(FORMULATIONS)):
    for j in range(len(BACKENDS)):
        v = grid[i, j]
        ax2.text(j, i, f"{v:.0f}%", ha="center", va="center",
                  color="white" if v > grid.max() * 0.5 else "black", fontsize=10)
ax2.set_title("Degraded ticks (% of 1250) -- full grid", fontsize=11)
fig2.colorbar(im, ax=ax2, label="% ticks degraded", fraction=0.046, pad=0.04)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "singularity_grid_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

print("Wrote:")
print(" ", OUT_DIR / "singularity_explicit_backends.png")
print(" ", OUT_DIR / "singularity_grid_heatmap.png")
print(" ", OUT_DIR / "singularity_summary.json")

# Print the table for direct pasting into the MD report.
print()
print(f"{'formulation':11s} {'backend':10s} {'sig_min':>8s} {'err_max':>8s} {'err_mean':>9s} {'tau_max':>8s} {'qdd_max':>10s} {'degraded':>10s}")
for r in summary:
    print(f"{r['form']:11s} {r['backend']:10s} {r['sigma_min_min']:8.4f} {r['err_max']:8.4f} {r['err_mean']:9.5f} {r['tau_max_abs_max']:8.2f} {r['qddot_norm_max']:10.1f} {r['n_degraded']:5d}/{r['n_total']}")
