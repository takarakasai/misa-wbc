//! The convenience entry point: hand a priority-ordered list of tasks
//! to [`solve`] and get back the decision vector, without wiring the
//! [`HoQp`](crate::HoQp) chain by hand.
//!
//! Two things are switchable through [`SolveConfig`]:
//!
//! - **The HQP strategy** ([`HqpStrategy`]) — how the priority hierarchy
//!   is resolved. Today: [`HqpStrategy::NullSpace`] (the Kim-2014
//!   null-space cascade). Future strategies (a weighted single QP, an
//!   equality-only pseudo-inverse eHQP) slot in as new variants without
//!   touching call sites.
//! - **The QP backend** ([`crate::qp::QpSolver`]) — which solver runs
//!   each inner QP: the built-in dense active-set method or Clarabel.
//!
//! This mirrors OpenSoT's two-level split (a `Solver` front-end over a
//! `BackEnd` QP engine) and makes A/B comparisons — strategy vs
//! strategy, backend vs backend — a config change, not a rewrite.

use nalgebra::DVector;

use crate::ho_qp::{HoQp, WarmStart};
use crate::qp::{QpConfig, QpSolver, QpStatus};
use crate::task::Task;

/// Which hierarchical-QP strategy resolves the priority stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HqpStrategy {
    /// Kim-2014 null-space cascade: each level is solved in the null
    /// space of all higher-priority equalities, with slack variables
    /// relaxing higher inequalities. Strict lexicographic priority.
    NullSpace,
    // Future: WeightedQp (single QP, priorities as weights — fast,
    // approximate), EHqp (equality-only pseudo-inverse), ...
}

/// Configuration for [`solve`]: the strategy, the inner-QP backend, and
/// the numerical knobs.
#[derive(Clone, Debug)]
pub struct SolveConfig {
    /// How the priority hierarchy is resolved.
    pub strategy: HqpStrategy,
    /// Which QP solver runs each inner problem.
    pub backend: QpSolver,
    /// Max inner-solver iterations.
    pub max_iters: usize,
    /// Constraint-feasibility tolerance.
    pub feasibility_tol: f64,
    /// Step / multiplier optimality tolerance.
    pub optimality_tol: f64,
    /// Proximal warm-start weight for [`solve_warm`]. `> 0` biases each
    /// tick's optimum toward the previous solution to damp jitter; `0`
    /// (default) is a cold solve.
    pub prox_weight: f64,
}

impl Default for SolveConfig {
    fn default() -> Self {
        // Mirror the QP defaults, but pick Clarabel: the HoQP inner
        // problems are equality + inequality + prox, which the IPM
        // handles more robustly than the active-set method.
        let qp = QpConfig::default();
        Self {
            strategy: HqpStrategy::NullSpace,
            backend: QpSolver::Clarabel,
            max_iters: qp.max_iters,
            feasibility_tol: qp.feasibility_tol,
            optimality_tol: qp.optimality_tol,
            prox_weight: 0.0,
        }
    }
}

impl SolveConfig {
    /// The inner-QP config this strategy hands to each level.
    fn qp_cfg(&self) -> QpConfig {
        QpConfig {
            solver: self.backend,
            max_iters: self.max_iters,
            feasibility_tol: self.feasibility_tol,
            optimality_tol: self.optimality_tol,
            // prox is applied per-level via the warm-start projection.
            prox_weight: 0.0,
        }
    }
}

/// Outcome of a solve: which levels (if any) failed to reach optimality.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SolveStatus {
    /// Every level's inner QP reached optimality.
    Optimal,
    /// At least one level degraded; carries the first offending
    /// `(level_index, status)`. The solution is still returned (the
    /// cascade holds the last good `x` for a failed level), so a host
    /// can decide whether to command it or hold.
    Degraded { level: usize, status: QpStatus },
}

/// A solved hierarchy.
#[derive(Clone, Debug)]
pub struct Solution {
    /// The global decision vector `x`.
    pub x: DVector<f64>,
    /// Per-solve status (optimal, or the first degraded level).
    pub status: SolveStatus,
    /// The full-space `x`, retained for warm-starting the next tick via
    /// [`solve_warm`]. (Same values as [`Solution::x`]; named to make
    /// the round-trip intent explicit at call sites.)
    pub warm_anchor: DVector<f64>,
}

/// Errors that prevent a solve from running at all (as opposed to a
/// degraded-but-returned solution, which is [`SolveStatus::Degraded`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WbcError {
    /// The level list was empty — nothing to solve.
    NoLevels,
    /// Two levels disagree on the decision-vector size.
    DimMismatch { level: usize, expected: usize, found: usize },
}

impl std::fmt::Display for WbcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WbcError::NoLevels => write!(f, "solve: no priority levels given"),
            WbcError::DimMismatch { level, expected, found } => write!(
                f,
                "solve: level {level} has n_decision {found}, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for WbcError {}

/// Solve a priority-ordered task list (index 0 = highest priority).
/// Cold start — no warm anchor.
pub fn solve(levels: &[Task], cfg: &SolveConfig) -> Result<Solution, WbcError> {
    solve_warm(levels, cfg, None)
}

/// Solve with an optional warm-start anchor: pass the previous tick's
/// [`Solution::warm_anchor`] to bias each level's inner QP toward it
/// (weighted by [`SolveConfig::prox_weight`]) and damp tick-to-tick
/// jitter. `None` (or `prox_weight == 0`) is a cold solve.
pub fn solve_warm(
    levels: &[Task],
    cfg: &SolveConfig,
    warm_anchor: Option<&DVector<f64>>,
) -> Result<Solution, WbcError> {
    if levels.is_empty() {
        return Err(WbcError::NoLevels);
    }
    // Validate the shared decision size (ignoring empty-task levels,
    // whose n_decision is 0 by convention).
    let n = levels
        .iter()
        .map(Task::n_decision)
        .find(|&d| d > 0)
        .unwrap_or(0);
    for (i, t) in levels.iter().enumerate() {
        let d = t.n_decision();
        if d != 0 && d != n {
            return Err(WbcError::DimMismatch { level: i, expected: n, found: d });
        }
    }

    match cfg.strategy {
        HqpStrategy::NullSpace => Ok(solve_null_space(levels, cfg, warm_anchor)),
    }
}

/// The Kim-2014 null-space cascade: chain [`HoQp`] levels, threading the
/// warm anchor and collecting the first degraded level.
fn solve_null_space(
    levels: &[Task],
    cfg: &SolveConfig,
    warm_anchor: Option<&DVector<f64>>,
) -> Solution {
    let qp_cfg = cfg.qp_cfg();
    let warm = WarmStart { x_prev: warm_anchor, prox_weight: cfg.prox_weight };

    let mut prev: Option<HoQp> = None;
    let mut degraded: Option<(usize, QpStatus)> = None;
    for (i, t) in levels.iter().enumerate() {
        let hqp = HoQp::new_with_cfg(t.clone(), prev.as_ref(), &warm, &qp_cfg);
        if degraded.is_none() && hqp.status() != QpStatus::Optimal {
            degraded = Some((i, hqp.status()));
        }
        prev = Some(hqp);
    }

    let last = prev.expect("non-empty checked by caller");
    let x = last.solution().clone();
    let status = match degraded {
        None => SolveStatus::Optimal,
        Some((level, status)) => SolveStatus::Degraded { level, status },
    };
    Solution { warm_anchor: x.clone(), x, status }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    /// x = [a(1); b(1)];  prio0: a = 1 (eq) + b ≤ 3 (ineq);  prio1: b = 2.
    fn levels() -> Vec<Task> {
        let p0 = Task::equality(
            DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
            DVector::from_vec(vec![1.0]),
        ) + Task::inequality(
            DMatrix::from_row_slice(1, 2, &[0.0, 1.0]),
            DVector::from_vec(vec![3.0]),
        );
        let p1 = Task::equality(
            DMatrix::from_row_slice(1, 2, &[0.0, 1.0]),
            DVector::from_vec(vec![2.0]),
        );
        vec![p0, p1]
    }

    #[test]
    fn empty_levels_errors() {
        assert_eq!(solve(&[], &SolveConfig::default()).unwrap_err(), WbcError::NoLevels);
    }

    #[test]
    fn dim_mismatch_errors() {
        let bad = vec![
            Task::equality(DMatrix::zeros(1, 2), DVector::zeros(1)),
            Task::equality(DMatrix::zeros(1, 3), DVector::zeros(1)),
        ];
        assert!(matches!(
            solve(&bad, &SolveConfig::default()),
            Err(WbcError::DimMismatch { level: 1, expected: 2, found: 3 })
        ));
    }

    #[cfg(feature = "clarabel")]
    #[test]
    fn solves_and_respects_priority() {
        let sol = solve(&levels(), &SolveConfig::default()).unwrap();
        assert_eq!(sol.status, SolveStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "a should track prio-0 eq");
        assert!((sol.x[1] - 2.0).abs() < 1e-6, "b should track prio-1 eq");
    }

    #[cfg(feature = "clarabel")]
    #[test]
    fn backend_is_switchable() {
        // Same problem, active-set backend — should reach the same answer.
        let cfg = SolveConfig { backend: QpSolver::ActiveSet, ..Default::default() };
        let sol = solve(&levels(), &cfg).unwrap();
        assert!((sol.x[0] - 1.0).abs() < 1e-5);
        assert!((sol.x[1] - 2.0).abs() < 1e-5);
    }

    #[cfg(feature = "clarabel")]
    #[test]
    fn warm_anchor_round_trips() {
        let cfg = SolveConfig { prox_weight: 1e-3, ..Default::default() };
        let s1 = solve(&levels(), &cfg).unwrap();
        let s2 = solve_warm(&levels(), &cfg, Some(&s1.warm_anchor)).unwrap();
        assert!((&s1.x - &s2.x).norm() < 1e-4);
    }
}
