//! End-to-end: assemble a realistic legged WBC stack from the `tasks`
//! catalogue over a `VarLayout`, solve it, and check the physics holds.
//!
//! This is the shape a host (a quadruped / arm controller) uses:
//! declare `x = [q̈; f; τ]`, build the equation of motion + contact +
//! tracking + friction + torque-limit tasks from its dynamics matrices,
//! and hand the priority stack to `solve`. Here the matrices are a small
//! synthetic-but-consistent system so the test needs no robot model.

#![cfg(feature = "clarabel")]

use misa_wbc::tasks;
use misa_wbc::{solve, SolveConfig, SolveStatus, VarLayout};
use nalgebra::{DMatrix, DVector};

/// A tiny consistent floating-base system: nv = 8 (6 base + 2 actuated),
/// one 3-D point contact, na = 2. `M = I`, gravity-only `h`, a contact
/// Jacobian whose base block is identity (so the foot senses base motion).
fn toy_system() -> (usize, usize, usize, DMatrix<f64>, DVector<f64>, DMatrix<f64>) {
    let (nv, nc, na) = (8usize, 1usize, 2usize);
    let mass = DMatrix::<f64>::identity(nv, nv);
    // gravity term on the base z row and both joints.
    let mut h = DVector::zeros(nv);
    h[2] = 9.81; // base vertical
    h[6] = 0.5;
    h[7] = -0.5;
    // contact linear Jacobian (3 × nv): identity on the base translation
    // block, small coupling to the joints.
    let mut jc = DMatrix::zeros(3 * nc, nv);
    for i in 0..3 {
        jc[(i, i)] = 1.0;
    }
    jc[(0, 6)] = 0.1;
    jc[(1, 7)] = 0.1;
    (nv, nc, na, mass, h, jc)
}

#[test]
fn assembled_stack_solves_and_satisfies_physics() {
    let (nv, nc, na, mass, h, jc) = toy_system();
    let vars = VarLayout::builder()
        .add("qddot", nv)
        .add("f", 3 * nc)
        .add("tau", na)
        .build();
    let q = vars.var("qddot");
    let f = vars.var("f");
    let tau = vars.var("tau");

    // dj_v = 0 (rest instant).
    let dj_v = DVector::zeros(3 * nc);

    // Priority 0 (hard physics): EoM + stance contact holds still +
    // friction cone + torque limits.
    let p0 = tasks::equation_of_motion(&q, &f, &tau, &mass, &h, &jc)
        + tasks::zero_contact_acceleration(&q, &jc, &dj_v)
        + tasks::friction_pyramid(&f, 0.7)
        + tasks::box_bound(&tau, &DVector::from_vec(vec![40.0, 40.0]));

    // Priority 1 (soft tracking): hold base still (q̈_base ≈ 0) via a
    // Cartesian-acceleration task on the base translation, and keep the
    // contact force near a nominal downward push.
    let j_base = {
        let mut m = DMatrix::zeros(3, nv);
        for i in 0..3 {
            m[(i, i)] = 1.0;
        }
        m
    };
    let p1 = tasks::cartesian_acceleration(&q, &j_base, &DVector::zeros(3), &DVector::zeros(3))
        + tasks::regularize(&f, &DVector::from_vec(vec![0.0, 0.0, 5.0]));

    let sol = solve(&[p0, p1], &SolveConfig::default()).expect("solve");
    assert_eq!(sol.status, SolveStatus::Optimal);

    let x = &sol.x;
    let qddot = q.extract(x);
    let force = f.extract(x);
    let torque = tau.extract(x);

    // 1. EoM holds:  M·q̈ − Jcᵀ·f − Sᵀ·τ + h ≈ 0.
    let mut s_t = DMatrix::zeros(nv, na);
    for i in 0..na {
        s_t[(nv - na + i, i)] = 1.0;
    }
    let eom = &mass * &qddot - jc.transpose() * &force - &s_t * &torque + &h;
    assert!(eom.norm() < 1e-5, "EoM violated: {}", eom.norm());

    // 2. Stance foot doesn't accelerate:  Jc·q̈ ≈ 0.
    assert!((&jc * &qddot).norm() < 1e-5, "contact accelerates");

    // 3. Friction cone respected:  |fx|,|fy| ≤ μ·fz, fz ≥ 0.
    assert!(force[2] >= -1e-6, "pulling force");
    assert!(force[0].abs() <= 0.7 * force[2] + 1e-5, "fx outside cone");
    assert!(force[1].abs() <= 0.7 * force[2] + 1e-5, "fy outside cone");

    // 4. Torque within limits.
    assert!(torque[0].abs() <= 40.0 + 1e-5 && torque[1].abs() <= 40.0 + 1e-5);
}
