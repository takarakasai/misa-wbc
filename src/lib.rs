//! # misa-wbc — model-agnostic whole-body control
//!
//! A hierarchical whole-body-control (WBC) library built around a
//! **Hierarchical Quadratic Program** (HoQP) and, on top of it, an
//! [OpenSoT]-style vocabulary of tasks, constraints and affine
//! optimization variables.
//!
//! [OpenSoT]: https://github.com/ADVRHumanoids/OpenSoT
//!
//! ## Boundary: this crate does not know your robot
//!
//! misa-wbc computes joint accelerations / contact forces / torques by
//! solving a prioritized QP over **matrices you hand it** — the mass
//! matrix `M`, nonlinear term `h`, contact Jacobians `Jc`, their bias
//! `J̇·v`, and so on. It never touches URDF, meshes, or a kinematics
//! engine. Consumers compute those matrices with whatever rigid-body
//! dynamics library they already have (e.g. `misarta`) and pass them
//! in. This mirrors OpenSoT's split (`iHQP` = the solver, the robot
//! model is an external `ModelInterface`) and keeps the dependency
//! surface to `nalgebra` (+ optional [Clarabel] for the interior-point
//! QP backend).
//!
//! [Clarabel]: https://clarabel.org
//!
//! ## Layers
//!
//! - [`qp`] — the internal dense QP solver ([`qp::solve_qp`]): a
//!   built-in active-set method plus an optional Clarabel backend,
//!   with proximal warm-start. Clarabel is the default backend; the
//!   built-in active-set solver needs no extra dependency.
//! - [`task`] — the elementary [`Task`]: a soft equality `A·x = b`
//!   (least-squares cost) plus a hard inequality `D·x ≤ f`, combined
//!   with `+` at one priority level.
//! - [`ho_qp`] — the [`HoQp`] hierarchical solver: each task is solved
//!   in the null space of all higher-priority tasks' equalities.
//! - [`dims`] — decision-vector bookkeeping (`x = [q̈; f; τ]`) for the
//!   common legged layout. The general variable-layout / affine
//!   vocabulary (arbitrary named variables) lands next (Phase 1).
//!
//! ## Status
//!
//! Phase 0: the HoQP core is carried over verbatim from the
//! (validated) quadruped-gait WBC and made model-agnostic. The
//! OpenSoT-style affine-variable layer and the generic task / constraint
//! catalogue (Cartesian acceleration, friction cone, torque limits,
//! CBF joint limits, force distribution) follow. See the design study in
//! the articara repo (`ref/opensot.md`).

pub mod dims;
pub mod ho_qp;
pub mod qp;
pub mod task;

pub use dims::WbcDims;
pub use ho_qp::{HoQp, WarmStart};
pub use qp::{solve_qp, QpConfig, QpSolution, QpSolver, QpStatus};
pub use task::Task;
