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
//! - [`solve`] — the convenience entry point ([`solve`] / [`solve_warm`]):
//!   hand it a priority-ordered task list and a [`SolveConfig`], which
//!   selects the HQP strategy ([`HqpStrategy`]) and the QP backend
//!   ([`QpSolver`]) so strategy / backend are a config switch, not a
//!   rewrite.
//! - [`tasks`] — a catalogue of ready-made WBC tasks (equation of
//!   motion, Cartesian acceleration, friction pyramid, box limits,
//!   tracking / regularisation) built over the affine layer from the
//!   `M / h / J / J̇v` matrices a dynamics engine produces.
//! - [`affine`] — the OpenSoT-style variable vocabulary:
//!   [`VarLayout`] declares a named decision-vector layout (contact
//!   count is runtime, not fixed) and [`Affine`] (`y = M·x + q`)
//!   composes task residuals symbolically. [`Task::soft_eq`] /
//!   [`Task::le`] / [`Task::ge`] / [`Task::in_range`] turn an affine
//!   into a task.
//! - [`dims`] — fixed-layout bookkeeping (`x = [q̈; f; τ]`) for the
//!   common legged case; a thin alternative to `VarLayout`.
//!
//! ## Status
//!
//! Phase 0: the HoQP core carried over from the (validated)
//! quadruped-gait WBC and made model-agnostic. Phase 1: the
//! OpenSoT-style affine-variable layer ([`affine`]). Next: the generic
//! task / constraint catalogue (Cartesian acceleration, friction cone,
//! torque limits, CBF joint limits, force distribution). See the design
//! study in the articara repo (`ref/opensot.md`).

pub mod affine;
pub mod dims;
pub mod ho_qp;
pub mod qp;
pub mod solve;
pub mod task;
pub mod tasks;

pub use affine::{Affine, Var, VarLayout, VarLayoutBuilder};
pub use dims::WbcDims;
pub use ho_qp::{HoQp, WarmStart};
pub use qp::{solve_qp, QpConfig, QpSolution, QpSolver, QpStatus};
pub use solve::{
    solve, solve_warm, HqpStrategy, Solution, SolveConfig, SolveStatus, WbcError,
};
pub use task::Task;
