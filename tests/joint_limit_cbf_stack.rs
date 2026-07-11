//! End-to-end: does [`tasks::joint_limit_cbf`] actually keep a joint
//! inside its position/velocity limits under closed-loop simulation?
//!
//! Unit tests in `src/tasks.rs` check the constraint's math in
//! isolation (one tick, hand-picked state). This test is the thing
//! that matters in practice: a runaway low-priority tracking task
//! that always demands maximum acceleration toward the limit, with
//! the CBF at priority 0 as the only thing standing in its way, run
//! through many simulated ticks with real (semi-implicit Euler)
//! integration. If the CBF's math or its priority wiring were wrong,
//! this is where it would show up as `q` walking straight through
//! `q_max`.

#![cfg(feature = "clarabel")]

use misa_wbc::tasks::{self, JointLimitCbf};
use misa_wbc::{solve, SolveConfig, SolveStatus, VarLayout};
use nalgebra::DVector;

/// 1-DOF double integrator (`q̈ = u`) under closed-loop CBF safety +
/// an adversarial tracking task that always asks for maximum
/// acceleration toward `q_max`. Returns the full position trace.
fn simulate(limits: &JointLimitCbf, ticks: usize, dt: f64) -> Vec<f64> {
    let vars = VarLayout::builder().add("qddot", 1).build();
    let qddot = vars.var("qddot");

    let mut q = DVector::from_vec(vec![0.0]);
    let mut v = DVector::from_vec(vec![0.0]);
    let mut trace = Vec::with_capacity(ticks + 1);
    trace.push(q[0]);

    for _ in 0..ticks {
        // Priority 0: the safety barrier, rebuilt from the current state.
        let p0 = tasks::joint_limit_cbf(&qddot, &q, &v, limits);
        // Priority 1: a runaway controller that always wants to
        // accelerate as hard as possible toward q_max. Deliberately
        // asks for far more than a_max so it never "gives up" on its
        // own — only the CBF can stop it.
        let p1 = tasks::regularize(&qddot, &DVector::from_vec(vec![1000.0]));

        let sol = solve(&[p0, p1], &SolveConfig::default()).expect("solve");
        assert_eq!(sol.status, SolveStatus::Optimal, "solver degraded mid-simulation");

        let a = sol.x[0];
        // Semi-implicit Euler: update v first, then use the new v for q
        // (more stable than explicit Euler for a hard barrier at the edge).
        v[0] += a * dt;
        q[0] += v[0] * dt;
        trace.push(q[0]);
    }
    trace
}

#[test]
fn cbf_keeps_a_runaway_joint_inside_its_position_limit() {
    let limits = JointLimitCbf {
        q_min: DVector::from_vec(vec![-1.0]),
        q_max: DVector::from_vec(vec![1.0]),
        v_max: DVector::from_vec(vec![5.0]),
        a_max: DVector::from_vec(vec![50.0]),
        alpha1: DVector::from_vec(vec![8.0]),
        alpha2: DVector::from_vec(vec![8.0]),
        alpha3: DVector::from_vec(vec![8.0]),
    };
    let trace = simulate(&limits, 2000, 0.002); // 4 s of simulated motion

    let max_q = trace.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_q <= limits.q_max[0] + 1e-3,
        "runaway controller broke through q_max: reached {max_q}, limit {}",
        limits.q_max[0]
    );

    // The controller is genuinely adversarial (always demanding the
    // max toward q_max), so a working barrier should also make the
    // joint settle NEAR the limit, not just "never technically cross
    // it while sitting far away" — that would indicate the CBF is
    // over-conservative rather than correctly tight.
    let final_q = *trace.last().unwrap();
    assert!(
        final_q > limits.q_max[0] - 0.05,
        "expected the joint to settle near q_max under constant pressure, got {final_q}"
    );
}

#[test]
fn cbf_keeps_a_runaway_joint_inside_its_velocity_limit() {
    // Same setup but with a much higher position limit (so position
    // never binds) and a tight v_max — only the 1st-order velocity
    // barrier should be doing the work here.
    let limits = JointLimitCbf {
        q_min: DVector::from_vec(vec![-1000.0]),
        q_max: DVector::from_vec(vec![1000.0]),
        v_max: DVector::from_vec(vec![0.5]),
        a_max: DVector::from_vec(vec![50.0]),
        alpha1: DVector::from_vec(vec![8.0]),
        alpha2: DVector::from_vec(vec![8.0]),
        alpha3: DVector::from_vec(vec![8.0]),
    };
    let vars = VarLayout::builder().add("qddot", 1).build();
    let qddot = vars.var("qddot");
    let mut q = DVector::from_vec(vec![0.0]);
    let mut v = DVector::from_vec(vec![0.0]);
    let dt = 0.002;
    let mut max_v = 0.0_f64;

    for _ in 0..2000 {
        let p0 = tasks::joint_limit_cbf(&qddot, &q, &v, &limits);
        let p1 = tasks::regularize(&qddot, &DVector::from_vec(vec![1000.0]));
        let sol = solve(&[p0, p1], &SolveConfig::default()).expect("solve");
        assert_eq!(sol.status, SolveStatus::Optimal);
        let a = sol.x[0];
        v[0] += a * dt;
        q[0] += v[0] * dt;
        max_v = max_v.max(v[0]);
    }

    assert!(
        max_v <= limits.v_max[0] + 1e-3,
        "runaway controller broke through v_max: reached {max_v}, limit {}",
        limits.v_max[0]
    );
}
