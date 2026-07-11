//! The convenience entry point: hand a priority-ordered list of tasks
//! to [`solve`] and get back the decision vector, without wiring the
//! [`HoQp`] chain by hand.
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
use crate::qp::{QpConfig, QpSolver, QpStatus, QpWorkspace};
use crate::task::Task;

/// Which hierarchical-QP strategy resolves the priority stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HqpStrategy {
    /// Kim-2014 null-space cascade: each level is solved in the null
    /// space of all higher-priority equalities, with slack variables
    /// relaxing higher inequalities. Strict lexicographic priority.
    NullSpace,
    /// GID-style greedy force-budget cascade: each level solves a small
    /// weighted least-squares QP for an *increment* on the committed
    /// solution, then commits it. Inequalities accumulate downward (a
    /// level's constraints bind every later level), so torque boxes at
    /// level 0 become the shrinking budget of lower levels. Priority is
    /// greedy, **not** lexicographic: a lower level may degrade an upper
    /// task if the constraints leave it room (this is GID's documented
    /// behaviour, useful for equivalence studies against it).
    ForceBudgetCascade,
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

    solve_dispatch(levels, cfg, warm_anchor, n, None)
}

fn solve_dispatch(
    levels: &[Task],
    cfg: &SolveConfig,
    warm_anchor: Option<&DVector<f64>>,
    n: usize,
    workspaces: Option<&mut [QpWorkspace]>,
) -> Result<Solution, WbcError> {
    match cfg.strategy {
        HqpStrategy::NullSpace => Ok(solve_null_space(levels, cfg, warm_anchor, workspaces)),
        HqpStrategy::ForceBudgetCascade => Ok(solve_force_budget(levels, cfg, n, workspaces)),
    }
}

/// A persistent solver session: one [`QpWorkspace`] per priority level,
/// so the active-set backend warm-starts each level's inner QP from the
/// previous tick's solution and working set (the qpOASES online-active-
/// set pattern carried through the whole hierarchy). With the Clarabel
/// backend it behaves exactly like the free [`solve`] function.
///
/// Keep one `Solver` alive for the lifetime of a controller and call
/// [`Solver::solve`] every tick:
///
/// ```
/// # #[cfg(feature = "clarabel")] {
/// use misa_wbc::{Solver, SolveConfig, Task};
/// use nalgebra::{DMatrix, DVector};
///
/// let mut solver = Solver::new();
/// let cfg = SolveConfig::default();
/// let level = Task::equality(DMatrix::identity(2, 2), DVector::from_vec(vec![1.0, 2.0]));
/// for _tick in 0..3 {
///     let sol = solver.solve(&[level.clone()], &cfg).unwrap();
///     assert!((sol.x[0] - 1.0).abs() < 1e-6);
/// }
/// # }
/// ```
///
/// If the level structure changes between ticks (different level count
/// or inner dimensions — e.g. a contact appears), the affected
/// workspaces simply fall back to a cold start for that tick; no reset
/// is required for correctness.
#[derive(Debug, Default)]
pub struct Solver {
    workspaces: Vec<QpWorkspace>,
    /// The previous tick's solution, fed back as the proximal anchor
    /// when [`SolveConfig::prox_weight`] > 0.
    anchor: Option<DVector<f64>>,
}

impl Solver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop all warm-start state — the next solve is fully cold.
    pub fn reset(&mut self) {
        self.workspaces.clear();
        self.anchor = None;
    }

    /// [`solve`] with this session's warm-start state. The stored
    /// solution anchor is used for the proximal term automatically when
    /// `cfg.prox_weight > 0`.
    pub fn solve(&mut self, levels: &[Task], cfg: &SolveConfig) -> Result<Solution, WbcError> {
        if levels.is_empty() {
            return Err(WbcError::NoLevels);
        }
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
        self.workspaces.resize_with(levels.len(), QpWorkspace::default);

        let anchor = if cfg.prox_weight > 0.0 { self.anchor.clone() } else { None };
        let sol = solve_dispatch(
            levels,
            cfg,
            anchor.as_ref(),
            n,
            Some(&mut self.workspaces[..]),
        )?;
        self.anchor = Some(sol.warm_anchor.clone());
        Ok(sol)
    }
}

/// The Kim-2014 null-space cascade: chain [`HoQp`] levels, threading the
/// warm anchor and collecting the first degraded level.
fn solve_null_space(
    levels: &[Task],
    cfg: &SolveConfig,
    warm_anchor: Option<&DVector<f64>>,
    mut workspaces: Option<&mut [QpWorkspace]>,
) -> Solution {
    let qp_cfg = cfg.qp_cfg();
    let warm = WarmStart { x_prev: warm_anchor, prox_weight: cfg.prox_weight };

    let mut prev: Option<HoQp> = None;
    let mut degraded: Option<(usize, QpStatus)> = None;
    for (i, t) in levels.iter().enumerate() {
        let ws_i = workspaces.as_deref_mut().map(|w| &mut w[i]);
        let hqp = HoQp::new_with_cfg_ws(t.clone(), prev.as_ref(), &warm, &qp_cfg, ws_i);
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

/// The GID-style force-budget cascade (see [`HqpStrategy::ForceBudgetCascade`]).
///
/// Per level `k`, with committed solution `x_c` and the inequality rows
/// accumulated from levels `0..=k`:
///
/// ```text
///   min_δ  ‖A_k(x_c + δ) − b_k‖² + ρ·‖δ‖²
///   s.t.   D_{0..k}·(x_c + δ) ≤ f_{0..k}
/// ```
///
/// then `x_c ← x_c + δ`. The regularisation `ρ` is
/// [`SolveConfig::prox_weight`] with a `1e-8` floor (GID's tiny
/// minimum-effort term; it also makes each level's optimum unique). The
/// warm anchor is unused — the cascade always increments from `x_c`,
/// mirroring GID's committed `JointForce`. On a level failure the
/// increment is skipped (GID's use-last-force fallback) and the first
/// failure is reported as [`SolveStatus::Degraded`].
fn solve_force_budget(
    levels: &[Task],
    cfg: &SolveConfig,
    n: usize,
    mut workspaces: Option<&mut [QpWorkspace]>,
) -> Solution {
    use nalgebra::DMatrix;

    let qp_cfg = cfg.qp_cfg();
    let rho = cfg.prox_weight.max(1e-8);

    let mut x_c = DVector::<f64>::zeros(n);
    let mut stacked_d: Option<DMatrix<f64>> = None;
    let mut stacked_f: Option<DVector<f64>> = None;
    let mut degraded: Option<(usize, QpStatus)> = None;

    for (k, t) in levels.iter().enumerate() {
        // Accumulate this level's inequalities (they bind from here on).
        if t.n_iq() > 0 {
            stacked_d = Some(match stacked_d.take() {
                None => t.d.clone(),
                Some(d) => stack_rows(&d, &t.d),
            });
            stacked_f = Some(match stacked_f.take() {
                None => t.f.clone(),
                Some(f) => stack_vec(&f, &t.f),
            });
        }

        // H = A_kᵀA_k + ρI ;  g = A_kᵀ(A_k x_c − b_k)  (+ 0 from the ρ‖δ‖² center).
        let mut h = DMatrix::<f64>::identity(n, n) * rho;
        let mut g = DVector::<f64>::zeros(n);
        if t.n_eq() > 0 {
            h += t.a.transpose() * &t.a;
            g += t.a.transpose() * (&t.a * &x_c - &t.b);
        }

        // Shifted inequalities:  D·δ ≤ f − D·x_c.
        let shifted_f = stacked_f
            .as_ref()
            .map(|f| f - stacked_d.as_ref().expect("d/f stacked together") * &x_c);

        let ws_k = workspaces.as_deref_mut().map(|w| &mut w[k]);
        let sol = match ws_k {
            Some(ws) => crate::qp::solve_qp_warm(
                &h,
                &g,
                None,
                None,
                stacked_d.as_ref(),
                shifted_f.as_ref(),
                None,
                &qp_cfg,
                ws,
            ),
            None => crate::qp::solve_qp(
                &h,
                &g,
                None,
                None,
                stacked_d.as_ref(),
                shifted_f.as_ref(),
                None,
                &qp_cfg,
            ),
        };
        if sol.status == QpStatus::Optimal {
            x_c += &sol.x;
        } else if degraded.is_none() {
            // Skip the increment (use-last-force) and report.
            degraded = Some((k, sol.status));
        }
    }

    let status = match degraded {
        None => SolveStatus::Optimal,
        Some((level, status)) => SolveStatus::Degraded { level, status },
    };
    Solution { warm_anchor: x_c.clone(), x: x_c, status }
}

fn stack_rows(top: &nalgebra::DMatrix<f64>, bottom: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
    let mut out = nalgebra::DMatrix::zeros(top.nrows() + bottom.nrows(), top.ncols());
    out.rows_mut(0, top.nrows()).copy_from(top);
    out.rows_mut(top.nrows(), bottom.nrows()).copy_from(bottom);
    out
}

fn stack_vec(top: &DVector<f64>, bottom: &DVector<f64>) -> DVector<f64> {
    let mut out = DVector::zeros(top.len() + bottom.len());
    out.rows_mut(0, top.len()).copy_from(top);
    out.rows_mut(top.len(), bottom.len()).copy_from(bottom);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    /// x = [a(1); b(1)];  prio0: a = 1 (eq) + b ≤ 3 (ineq);  prio1: b = 2.
    #[cfg(feature = "clarabel")]
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
    #[cfg(feature = "clarabel")]
    #[test]
    fn session_matches_stateless_and_stays_optimal() {
        // ActiveSet backend + session: repeated ticks must stay optimal
        // and agree with the stateless solve (and Clarabel).
        let cfg = SolveConfig { backend: QpSolver::ActiveSet, ..Default::default() };
        let mut solver = Solver::new();
        let stateless = solve(&levels(), &SolveConfig::default()).unwrap();
        for _tick in 0..5 {
            let s = solver.solve(&levels(), &cfg).unwrap();
            assert_eq!(s.status, SolveStatus::Optimal);
            assert!((&s.x - &stateless.x).norm() < 1e-5, "session drifted");
        }
    }

    #[cfg(feature = "clarabel")]
    #[test]
    fn session_survives_level_structure_changes() {
        // Changing level count / dimensions between ticks must fall back
        // to a cold start gracefully, not corrupt the result.
        let cfg = SolveConfig::default();
        let mut solver = Solver::new();
        let _ = solver.solve(&levels(), &cfg).unwrap();

        // Different structure: one level, three variables.
        let other = vec![Task::equality(
            DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            DVector::from_vec(vec![3.0, 4.0]),
        )];
        let s = solver.solve(&other, &cfg).unwrap();
        assert_eq!(s.status, SolveStatus::Optimal);
        assert!((s.x[0] - 3.0).abs() < 1e-6 && (s.x[1] - 4.0).abs() < 1e-6);

        // And back again.
        let s = solver.solve(&levels(), &cfg).unwrap();
        assert!((s.x[0] - 1.0).abs() < 1e-6 && (s.x[1] - 2.0).abs() < 1e-6);
    }

}
