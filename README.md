# misa-wbc

[![CI](https://github.com/takarakasai/misa-wbc/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/takarakasai/misa-wbc/actions/workflows/ci.yml) [![coverage](https://codecov.io/gh/takarakasai/misa-wbc/graph/badge.svg)](https://codecov.io/gh/takarakasai/misa-wbc)

Model-agnostic **whole-body control** for robots: a hierarchical QP
(HoQP) core with an [OpenSoT](https://github.com/ADVRHumanoids/OpenSoT)-style
vocabulary of tasks, constraints and affine optimization variables.
Companion to the [`misarta`](https://github.com/takarakasai/misarta)
rigid-body-dynamics library — misarta computes the matrices, misa-wbc
solves the control QP.

## Boundary

misa-wbc solves a prioritized QP over the **matrices you hand it** — the
mass matrix `M`, nonlinear term `h`, contact Jacobians `Jc` and their
bias `J̇·v`. It never touches URDF, meshes, or a kinematics engine.
Compute those with whatever rigid-body-dynamics library you already have
(e.g. [`misarta`](https://github.com/takarakasai/misarta)) and pass them
in. This keeps the dependency surface at `nalgebra` (+ optional
[Clarabel](https://clarabel.org) for the interior-point QP backend), so
arm / legged / humanoid hosts can share one WBC without pulling in a
mesh / RBD stack.

## Status

The HoQP core is carried over from the (walking-validated)
quadruped-gait WBC and made model-agnostic, then extended:

- `qp` — internal dense QP solver (active-set + optional Clarabel,
  proximal warm-start), self-contained on nalgebra.
- `task` — the elementary `Task` (`A·x = b` soft equality + `D·x ≤ f`
  hard inequality), combined with `+` at one priority level.
- `ho_qp` — the `HoQp` hierarchical solver (each task solved in the null
  space of higher-priority equalities).
- `affine` — OpenSoT-style named variables (`VarLayout`) and affine
  expressions (`Affine`), so variable structure is runtime, not fixed.
- `tasks` — the generic catalogue: equation of motion, Cartesian
  acceleration, contact no-motion, friction pyramid, box limits,
  tracking / regularisation.
- `solve` — the entry point with switchable HQP strategy
  (`NullSpace` — strict lexicographic; `ForceBudgetCascade` — the
  GID-style greedy force-budget hierarchy) and QP backend.
- `dynamics` — **formulation switch**: the same task declarations run
  as `x = [q̈; f; τ]` (explicit), `x = [q̈; f]` (τ eliminated,
  OpenSoT-style) or `x = [τ; f]` (q̈ eliminated through `M⁻¹`,
  GID-style force space), for equal-footing comparisons across the
  three whole-body-control formulations.

**Next** — the ergonomic stack layer (named levels, per-task
achievement report), patch contacts (CoP box + torsional friction),
reference generators (PD / impedance), centroidal momentum task.
Quadruped-specific weights stay in quadruped-gait.

## Features

- `clarabel` (default) — Clarabel interior-point backend for the HoQP
  inner problems. `--no-default-features` leaves only the built-in
  active-set solver (enough for the standalone `qp` module).

## License

Apache-2.0
