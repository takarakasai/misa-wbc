# misa-wbc

Model-agnostic **whole-body control** for robots: a hierarchical QP
(HoQP) core with an [OpenSoT](https://github.com/ADVRHumanoids/OpenSoT)-style
vocabulary of tasks, constraints and affine optimization variables.

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

**Phase 0** — the HoQP core is carried over from the (walking-validated)
quadruped-gait WBC and made model-agnostic:

- `qp` — internal dense QP solver (active-set + optional Clarabel,
  proximal warm-start), self-contained on nalgebra.
- `task` — the elementary `Task` (`A·x = b` soft equality + `D·x ≤ f`
  hard inequality), combined with `+` at one priority level.
- `ho_qp` — the `HoQp` hierarchical solver (each task solved in the null
  space of higher-priority equalities).
- `dims` — decision-vector bookkeeping for `x = [q̈; f; τ]`.

**Next** — the OpenSoT-style affine-variable layer (arbitrary named
variables, so contact count and variable structure are runtime, not
fixed) and the generic task / constraint catalogue (Cartesian
acceleration, friction cone, torque limits, CBF joint limits, force
distribution). Quadruped-specific weights stay in quadruped-gait.

## Features

- `clarabel` (default) — Clarabel interior-point backend for the HoQP
  inner problems. `--no-default-features` leaves only the built-in
  active-set solver (enough for the standalone `qp` module).

## License

Apache-2.0
