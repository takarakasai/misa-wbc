//! OpenSoT-style correctness tests for the model-agnostic core.
//!
//! Ported in spirit from OpenSoT's `tests/solvers/TestBasicAlgebra.cpp`
//! (`testPinvVSQP`), `tests/utils/TestAutoStack.cpp`, and
//! `tests/solvers/TestiHQP.cpp`: **randomised problems checked against a
//! closed-form (pseudo-inverse) reference**. These pin that misa-wbc's
//! HoQP is a faithful hierarchical-least-squares / null-space-projected
//! pseudo-inverse — the mathematical contract the quadruped WBC it was
//! ported from relies on.
//!
//! Model-dependent task tests (Cartesian, CoM, friction cone, torque
//! limits — OpenSoT's `tests/tasks/*` and `tests/constraints/*`) live
//! with those tasks when they land (Phase 3+), since they need a robot
//! model to produce Jacobians.

use misa_wbc::{solve, HqpStrategy, SolveConfig, SolveStatus, Task, VarLayout, WbcError};
use nalgebra::{DMatrix, DVector};

// A fixed-seed LCG so the randomised checks are deterministic across
// runs (no rand dependency — this crate stays nalgebra-only).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_f64(&mut self) -> f64 {
        // Numerical Recipes LCG, mapped to [-1, 1).
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn matrix(&mut self, r: usize, c: usize) -> DMatrix<f64> {
        DMatrix::from_fn(r, c, |_, _| self.next_f64())
    }
    fn vector(&mut self, n: usize) -> DVector<f64> {
        DVector::from_fn(n, |_, _| self.next_f64())
    }
}

/// OpenSoT `testPinvVSQP`: for a random single equality task `A·x = b`,
/// the HoQP solution reaches the same **observable** `A·x` as the
/// pseudo-inverse reference `A·pinv(A)·b` (the best-reachable value).
/// Following OpenSoT, we compare the observable `A·x`, not the raw `x` —
/// under-determined systems have a null space, so the raw solution is
/// not unique and only the projection onto row-space is well-defined.
#[cfg(feature = "clarabel")]
#[test]
fn single_equality_matches_pseudo_inverse() {
    let cfg = SolveConfig::default();
    let mut rng = Lcg::new(0xC0FFEE);
    for _ in 0..20 {
        // Wide (under-determined), square, and tall (over-determined).
        for &(rows, cols) in &[(2usize, 4usize), (3, 3), (5, 3)] {
            let a = rng.matrix(rows, cols);
            let b = rng.vector(rows);
            let task = Task::equality(a.clone(), b.clone());

            let x_hqp = solve(std::slice::from_ref(&task), &cfg).unwrap().x;
            let x_pinv = a.clone().pseudo_inverse(1e-12).unwrap() * &b;

            // Observable (row-space projection) is the same.
            assert!(
                (&a * &x_hqp - &a * &x_pinv).norm() < 1e-5,
                "HoQP vs pinv observable differ ({rows}x{cols}): {:?} vs {:?}",
                &a * &x_hqp,
                &a * &x_pinv,
            );
            // Square / tall → also the least-squares residual matches.
            if rows >= cols {
                let r_hqp = (&a * &x_hqp - &b).norm();
                let r_pinv = (&a * &x_pinv - &b).norm();
                assert!((r_hqp - r_pinv).abs() < 1e-6, "residual norms differ");
            }
        }
    }
}

/// OpenSoT iHQP contract: in a two-level equality stack the upper task
/// is preserved exactly and the lower task is solved **in the upper's
/// null space**. With 2 upper rows, 2 lower rows and 5 variables the
/// upper leaves a 3-D null space, into which two independent lower
/// equations fit — so the lower task is reached exactly, *without*
/// disturbing the upper. Randomised so it isn't one hand-picked case.
///
/// (This is the observable, closed-form-free version of OpenSoT's
/// pseudo-inverse check: rather than reconstruct the nested projected
/// pseudo-inverse — which hinges on a numerically delicate null-space
/// basis — we assert the two properties that basis-independently define
/// a correct hierarchical solve: upper preserved, lower achieved.)
#[cfg(feature = "clarabel")]
#[test]
fn two_level_hierarchy_preserves_upper_and_achieves_lower() {
    let cfg = SolveConfig::default();
    let mut rng = Lcg::new(0x1234_5678);
    let n = 5;

    for _ in 0..20 {
        let a1 = rng.matrix(2, n);
        let b1 = rng.vector(2);
        let a2 = rng.matrix(2, n);
        let b2 = rng.vector(2);

        let levels = vec![
            Task::equality(a1.clone(), b1.clone()),
            Task::equality(a2.clone(), b2.clone()),
        ];
        let x = solve(&levels, &cfg).unwrap().x;

        // Priority: upper preserved exactly.
        assert!((&a1 * &x - &b1).norm() < 1e-5, "upper task disturbed");
        // Lower reached exactly (it fits in the upper's 3-D null space).
        assert!((&a2 * &x - &b2).norm() < 1e-5, "lower task not achieved");
    }
}

/// Companion to the above with the priority actually biting: a lower
/// task that *conflicts* with the upper is sacrificed, and lowering its
/// priority must never move the upper solution. Upper: x[0] = 1. Lower:
/// x[0] = 2 (direct conflict on the same coordinate).
#[cfg(feature = "clarabel")]
#[test]
fn conflicting_lower_yields_to_upper() {
    let cfg = SolveConfig::default();
    let a = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
    let upper = Task::equality(a.clone(), DVector::from_vec(vec![1.0]));
    let lower = Task::equality(a.clone(), DVector::from_vec(vec![2.0]));
    let x = solve(&[upper, lower], &cfg).unwrap().x;
    // Upper wins on the conflicted coordinate.
    assert!((x[0] - 1.0).abs() < 1e-5, "upper must win: x0 = {}", x[0]);
}

/// OpenSoT `TestAutoStack` operator semantics: `Task + Task` at one
/// priority level is a weighted least-squares of the concatenation, and
/// `weight` biases which residual wins when two conflict.
#[cfg(feature = "clarabel")]
#[test]
fn same_level_sum_and_weight_bias() {
    let cfg = SolveConfig::default();
    let vars = VarLayout::builder().add("x", 1).build();
    let x = vars.var("x");

    // Two conflicting scalar targets at one level: x ≈ 0 and x ≈ 10.
    let to_zero = Task::soft_eq(&(&x.affine() - &DVector::from_vec(vec![0.0])));
    let to_ten = Task::soft_eq(&(&x.affine() - &DVector::from_vec(vec![10.0])));

    // Equal weight → midpoint 5.
    let mid = solve(&[to_zero.clone() + to_ten.clone()], &cfg).unwrap().x[0];
    assert!((mid - 5.0).abs() < 1e-4, "equal-weight midpoint = {mid}");

    // Weight the "ten" target 9× → optimum at 10·9/(1+9) = 9.
    let biased = solve(&[to_zero + to_ten.weight(9.0)], &cfg).unwrap().x[0];
    assert!((biased - 9.0).abs() < 1e-4, "biased optimum = {biased}");
}

/// Contract: an over-constrained inequality stack is absorbed by the
/// HoQP's slack variables rather than failing — the inner QP stays
/// feasible (Optimal) and returns a compromise. This is the Kim-2014
/// design: inequalities at a level are relaxed by slack pushed into the
/// next level, so a single level never reports infeasible. (A genuine
/// `Degraded` needs the backend itself to fail — numerical, not
/// modelling.)
#[cfg(feature = "clarabel")]
#[test]
fn infeasible_inequalities_absorbed_by_slack() {
    // x ≤ −1  and  x ≥ +1  at one priority level: modelling-infeasible.
    let d_le = DMatrix::from_row_slice(1, 1, &[1.0]);
    let d_ge = DMatrix::from_row_slice(1, 1, &[-1.0]);
    let level = Task::inequality(d_le, DVector::from_vec(vec![-1.0]))
        + Task::inequality(d_ge, DVector::from_vec(vec![-1.0]));
    let sol = solve(&[level], &SolveConfig::default())
        .expect("slack keeps the inner QP feasible");
    assert_eq!(sol.status, SolveStatus::Optimal, "slack should absorb it");
    // The compromise sits between the two bounds (near 0 by symmetry).
    assert!(sol.x[0].abs() < 1.0 + 1e-6, "compromise x = {}", sol.x[0]);
}

/// Run-blocking errors (empty stack, dimension mismatch) surface as
/// `WbcError`, distinct from a degraded-but-solved outcome.
#[test]
fn run_blocking_errors() {
    assert_eq!(
        solve(&[], &SolveConfig::default()).unwrap_err(),
        WbcError::NoLevels,
    );
    let mixed = vec![
        Task::equality(DMatrix::zeros(1, 3), DVector::zeros(1)),
        Task::equality(DMatrix::zeros(1, 4), DVector::zeros(1)),
    ];
    assert!(matches!(
        solve(&mixed, &SolveConfig::default()).unwrap_err(),
        WbcError::DimMismatch { level: 1, .. },
    ));
}

/// The two backends must agree on a well-posed problem (OpenSoT keeps a
/// backend-parametrised suite; here we assert cross-backend agreement).
#[cfg(feature = "clarabel")]
#[test]
fn backends_agree() {
    use misa_wbc::QpSolver;
    let mut rng = Lcg::new(0xABCD);
    for _ in 0..10 {
        // Over-determined → unique least-squares solution, so both
        // backends must land on the same x (not just the same A·x).
        let a = rng.matrix(4, 2);
        let b = rng.vector(4);
        let levels = vec![Task::equality(a, b)];
        let x_clara = solve(
            &levels,
            &SolveConfig { backend: QpSolver::Clarabel, ..Default::default() },
        )
        .unwrap()
        .x;
        let x_as = solve(
            &levels,
            &SolveConfig { backend: QpSolver::ActiveSet, ..Default::default() },
        )
        .unwrap()
        .x;
        assert!(
            (&x_clara - &x_as).norm() < 1e-4,
            "backends differ: {:?} vs {:?}",
            x_clara,
            x_as,
        );
    }
}

/// Sanity: the default strategy is the null-space cascade (the only one
/// today). Pins the switch surface so adding a strategy is a visible
/// change here.
#[test]
fn default_strategy_is_null_space() {
    assert_eq!(SolveConfig::default().strategy, HqpStrategy::NullSpace);
}
