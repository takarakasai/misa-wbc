//! Cross-formulation equivalence: the same physical WBC problem, built
//! through [`Dynamics`] under all three formulations, must produce the
//! same physical solution (q̈, f, τ) — this is the "compare against
//! OpenSoT / GID on equal footing" infrastructure (D8).
//!
//! Also pins the GID-equivalence path: `ForceSpace` +
//! `ForceBudgetCascade` reproduces a hand-assembled GID-style
//! operational-space QP (`I = J·M⁻¹·[Sᵀ Jcᵀ]`) to solver tolerance,
//! and the budget cascade's greedy (non-lexicographic) semantics are
//! documented by contrast with the null-space strategy.

#![cfg(feature = "clarabel")]

use misa_wbc::qp::{solve_qp, QpConfig, QpSolver};
use misa_wbc::{
    solve, tasks, Dynamics, Formulation, HqpStrategy, SolveConfig, SolveStatus, Task,
};
use nalgebra::{DMatrix, DVector};

/// A small consistent floating-base system with a non-trivial SPD mass
/// matrix (coupling), gravity-like h, and one point contact.
fn toy() -> (usize, usize, DMatrix<f64>, DVector<f64>, DMatrix<f64>) {
    let (nv, na) = (8usize, 2usize);
    let l = DMatrix::from_fn(nv, 3, |i, j| ((i + 2 * j) as f64 * 0.37).sin());
    let mass = DMatrix::<f64>::identity(nv, nv) + 0.1 * (&l * l.transpose());
    let mut h = DVector::zeros(nv);
    h[2] = 9.81;
    h[6] = 0.5;
    h[7] = -0.5;
    let mut jc = DMatrix::zeros(3, nv);
    for i in 0..3 {
        jc[(i, i)] = 1.0;
    }
    jc[(0, 6)] = 0.2;
    jc[(1, 7)] = -0.1;
    (nv, na, mass, h, jc)
}

/// The same physical stack, declared once over whatever formulation the
/// [`Dynamics`] context carries: physics at priority 0, base + force
/// tracking at priority 1, minimum-motion regularisation at priority 2
/// (which pins the physical solution uniquely, so the formulations are
/// comparable point-wise, not just observable-wise).
fn stack(d: &Dynamics, nv: usize, na: usize, jc: &DMatrix<f64>) -> Vec<Task> {
    let f = d.forces();
    let dj_v = DVector::zeros(3);

    let mut p0 = tasks::zero_contact_acceleration(d.qddot(), jc, &dj_v)
        + tasks::friction_pyramid(&f, 0.7)
        + tasks::box_bound(d.tau(), &DVector::from_vec(vec![40.0; na]));
    if let Some(phys) = d.dynamics_task() {
        p0 = phys + p0;
    }

    let mut j_base = DMatrix::zeros(3, nv);
    for i in 0..3 {
        j_base[(i, i)] = 1.0;
    }
    let p1 = tasks::cartesian_acceleration(d.qddot(), &j_base, &DVector::zeros(3), &DVector::zeros(3))
        + tasks::regularize(&f, &DVector::from_vec(vec![0.0, 0.0, 5.0]));

    let p2 = tasks::track(d.qddot(), &DVector::zeros(nv))
        + tasks::track(d.tau(), &DVector::zeros(na));

    vec![p0, p1, p2]
}

/// All three formulations agree on the physical solution — and each
/// one's extracted triple satisfies the equation of motion.
#[test]
fn formulations_agree_on_the_physical_solution() {
    let (nv, na, mass, h, jc) = toy();
    let n_base = nv - na;
    let mut s_t = DMatrix::zeros(nv, na);
    for i in 0..na {
        s_t[(n_base + i, i)] = 1.0;
    }
    let cfg = SolveConfig::default();

    let mut results = Vec::new();
    for formulation in [Formulation::Explicit, Formulation::AccelSpace, Formulation::ForceSpace] {
        let d = Dynamics::new(formulation, &mass, &h, &jc, na);
        let sol = solve(&stack(&d, nv, na, &jc), &cfg).expect("solve");
        assert_eq!(sol.status, SolveStatus::Optimal, "{formulation:?} degraded");
        let e = d.extract(&sol.x);

        // Physics holds in every formulation.
        let eom = &mass * &e.qddot + &h - &s_t * &e.tau - jc.transpose() * &e.forces;
        assert!(eom.norm() < 1e-5, "{formulation:?}: EoM residual {}", eom.norm());
        // Contact holds still.
        assert!((&jc * &e.qddot).norm() < 1e-5, "{formulation:?}: contact accelerates");

        results.push((formulation, e));
    }

    // Pairwise point-wise agreement of the physical triple.
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let (fa, ea) = &results[i];
            let (fb, eb) = &results[j];
            assert!(
                (&ea.qddot - &eb.qddot).norm() < 1e-4,
                "q̈ differs: {fa:?} vs {fb:?}: {}",
                (&ea.qddot - &eb.qddot).norm()
            );
            assert!(
                (&ea.forces - &eb.forces).norm() < 1e-4,
                "f differs: {fa:?} vs {fb:?}"
            );
            assert!(
                (&ea.tau - &eb.tau).norm() < 1e-4,
                "τ differs: {fa:?} vs {fb:?}"
            );
        }
    }
}

/// GID equivalence: `ForceSpace` + `ForceBudgetCascade` reproduces the
/// hand-assembled GID operational-space QP
///
/// ```text
///   min_x ‖I·x − (a_ref − bias)‖² + ρ‖x‖²   s.t.  C·f ≤ 0
///   I = J·M⁻¹·[Sᵀ Jcᵀ],   bias = J̇v − J·M⁻¹·h
/// ```
///
/// to solver tolerance — the same QP built from misarta-style matrices
/// instead of GID's matrix-free unit-force propagation.
#[test]
fn force_space_budget_matches_hand_built_gid_qp() {
    let (nv, na, mass, h, jc) = toy();
    let n_base = nv - na;
    let rho = 1e-6;

    let j_task = DMatrix::from_fn(3, nv, |i, j| ((i * 5 + j) as f64 * 0.23).cos());
    let dj_v = DVector::from_vec(vec![0.1, -0.2, 0.3]);
    let aref = DVector::from_vec(vec![1.0, 0.5, -0.7]);

    // ── misa-wbc path ──
    let d = Dynamics::new(Formulation::ForceSpace, &mass, &h, &jc, na);
    let level = tasks::cartesian_acceleration(d.qddot(), &j_task, &dj_v, &aref)
        + tasks::friction_pyramid(&d.forces(), 0.7);
    let cfg = SolveConfig {
        strategy: HqpStrategy::ForceBudgetCascade,
        prox_weight: rho,
        ..Default::default()
    };
    let ours = solve(std::slice::from_ref(&level), &cfg).expect("solve").x;

    // ── hand-built GID QP over x = [τ; f] ──
    let chol = mass.clone().cholesky().unwrap();
    let mut s_t = DMatrix::zeros(nv, na);
    for i in 0..na {
        s_t[(n_base + i, i)] = 1.0;
    }
    let minv_st = chol.solve(&s_t);
    let minv_jct = chol.solve(&jc.transpose());
    let minv_h: DVector<f64> = chol.solve(&h);

    let n = na + 3;
    let mut minv_all = DMatrix::zeros(nv, n); // [M⁻¹Sᵀ | M⁻¹Jcᵀ]
    minv_all.columns_mut(0, na).copy_from(&minv_st);
    minv_all.columns_mut(na, 3).copy_from(&minv_jct);
    let i_mat = &j_task * &minv_all; // GID's operational-space inverse inertia
    let bias = &dj_v - &j_task * &minv_h; // J̇v + J·q̈(x=0)

    let target = &aref - &bias;
    let hqp = i_mat.transpose() * &i_mat + DMatrix::identity(n, n) * rho;
    let g = -(i_mat.transpose() * &target);
    // friction pyramid rows on the f block
    #[rustfmt::skip]
    let c = DMatrix::from_row_slice(5, 3, &[
        0.0,  0.0, -1.0,
        1.0,  0.0, -0.7,
       -1.0,  0.0, -0.7,
        0.0,  1.0, -0.7,
        0.0, -1.0, -0.7,
    ]);
    let mut d_iq = DMatrix::zeros(5, n);
    d_iq.columns_mut(na, 3).copy_from(&c);
    let f_iq = DVector::zeros(5);

    let manual = solve_qp(
        &hqp,
        &g,
        None,
        None,
        Some(&d_iq),
        Some(&f_iq),
        None,
        &QpConfig { solver: QpSolver::Clarabel, ..Default::default() },
    );
    assert!(
        (&ours - &manual.x).norm() < 1e-5,
        "GID-equivalent QP disagrees: {:?} vs {:?}",
        ours,
        manual.x
    );
}

/// The budget cascade is greedy, not lexicographic — the documented GID
/// semantics. A lower level may pull the solution as far as the
/// accumulated constraints allow, degrading the upper task; the
/// null-space strategy preserves the upper task exactly. Both are
/// correct per their contracts; this pins the difference.
#[test]
fn budget_cascade_is_greedy_null_space_is_strict() {
    // L0: x = 2 with |x| ≤ 3.  L1: x = 10.
    let l0 = Task::equality(
        DMatrix::from_row_slice(1, 1, &[1.0]),
        DVector::from_vec(vec![2.0]),
    ) + Task::inequality(
        DMatrix::from_row_slice(2, 1, &[1.0, -1.0]),
        DVector::from_vec(vec![3.0, 3.0]),
    );
    let l1 = Task::equality(
        DMatrix::from_row_slice(1, 1, &[1.0]),
        DVector::from_vec(vec![10.0]),
    );
    let levels = vec![l0, l1];

    let budget = SolveConfig {
        strategy: HqpStrategy::ForceBudgetCascade,
        ..Default::default()
    };
    let x_greedy = solve(&levels, &budget).unwrap().x[0];
    // Greedy: L1 pulls x from 2 up to the constraint boundary 3.
    assert!((x_greedy - 3.0).abs() < 1e-5, "greedy should hit the bound: {x_greedy}");

    let strict = SolveConfig::default(); // NullSpace
    let x_strict = solve(&levels, &strict).unwrap().x[0];
    // Lexicographic: the upper task's optimum x = 2 is preserved.
    assert!((x_strict - 2.0).abs() < 1e-5, "strict should preserve upper: {x_strict}");
}
