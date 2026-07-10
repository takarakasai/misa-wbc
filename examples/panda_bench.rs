//! Franka Panda formulation × strategy benchmark (run with `--release`).
//!
//! Same shape as `formulation_bench` (the Go2-scale synthetic problem)
//! but on a **real robot model**: the Panda URDF (inertials included)
//! is loaded through misarta — a dev-dependency only, the library
//! itself stays nalgebra-only — and `M`, `h`, the end-effector Jacobian
//! and its `J̇·v` bias come from real CRBA / RNEA / Jacobian
//! computations at a realistic configuration.
//!
//! A fixed-base arm exercises the formulation corners the quadruped
//! doesn't: `nc = 0` (no contact forces) and `n_base = 0`, so
//! `AccelSpace` eliminates τ *completely* (no dynamics task remains at
//! all) and `ForceSpace` is the pure GID inverse-dynamics
//! `q̈ = M⁻¹(τ − h)`. With no contact equalities to trample, the GID
//! budget cascade is also physically coherent here — the arm is the
//! friendly case for GID semantics.

use std::time::Instant;

use misa_wbc::qp::QpSolver;
use misa_wbc::{
    solve, tasks, Dynamics, Extracted, Formulation, HqpStrategy, SolveConfig, SolveStatus, Task,
};
use nalgebra::{DMatrix, DVector};

fn main() {
    // ── Load the real model ─────────────────────────────────────────
    let path = format!("{}/examples/models/panda.urdf", env!("CARGO_MANIFEST_DIR"));
    let imported = misarta_formats::urdf::import(std::path::Path::new(&path))
        .expect("import panda.urdf");
    let (model, _vis, _col) =
        misarta::native::build_model(&imported.file).expect("build panda model");
    let (nv, nq) = (model.nv, model.nq);
    assert_eq!(nq, nv, "fixed-base arm expected");
    let na = nv; // every joint actuated, no floating base

    // End-effector: the deepest arm joint (panda_joint7).
    let ee_idx = model
        .joints
        .iter()
        .position(|j| j.name == "panda_joint7")
        .expect("panda_joint7 in model");

    // Ready pose + a non-trivial joint velocity.
    let mut q = vec![0.0; nq];
    for (i, &qi) in [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8].iter().enumerate() {
        q[i] = qi;
    }
    let v: Vec<f64> = (0..nv).map(|i| 0.3 * ((i as f64) * 0.8).sin()).collect();

    // ── Real dynamics matrices for this tick ────────────────────────
    let mass = misarta::crba::crba(&model, &q);
    let h = misarta::rnea::nonlinear_effects(&model, &q, &v);
    let jc = DMatrix::<f64>::zeros(0, nv); // no contacts
    let j_ee = misarta::jacobian::compute_joint_jacobian(&model, &q, ee_idx);
    let dj_v_6 = misarta::jacobian::compute_jacobian_dot_times_v(&model, &q, &v, ee_idx);
    let dj_v = DVector::from_column_slice(dj_v_6.as_slice());

    // Desired EE acceleration ([ang; lin] rows): translate, keep level.
    let a_ref = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.5, -0.3]);

    // Panda joint torque limits (7 arm joints; finger joints get 20 N).
    let mut tau_max = DVector::from_element(na, 20.0);
    for (i, &t) in [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0].iter().enumerate() {
        tau_max[i] = t;
    }

    let stack = |d: &Dynamics| -> Vec<Task> {
        let mut p0 = tasks::box_bound(d.tau(), &tau_max);
        if let Some(phys) = d.dynamics_task() {
            p0 = phys + p0;
        }
        let p1 = tasks::cartesian_acceleration(d.qddot(), &j_ee, &dj_v, &a_ref);
        let p2 = tasks::track(d.qddot(), &DVector::zeros(nv))
            + tasks::track(d.tau(), &DVector::zeros(na)).weight(0.01);
        vec![p0, p1, p2]
    };

    let gauges = |e: &Extracted| -> (f64, f64, f64) {
        let eom = (&mass * &e.qddot + &h - &e.tau).norm(); // S = I (fully actuated)
        let ee = (&j_ee * &e.qddot + &dj_v - &a_ref).norm();
        (eom, ee, e.tau.amax())
    };

    // ── Sweep ───────────────────────────────────────────────────────
    let formulations = [Formulation::Explicit, Formulation::AccelSpace, Formulation::ForceSpace];
    let strategies =
        [(HqpStrategy::NullSpace, "NullSpace"), (HqpStrategy::ForceBudgetCascade, "ForceBudget")];
    let backends = [(QpSolver::Clarabel, "Clarabel"), (QpSolver::ActiveSet, "ActiveSet")];

    const WARMUP: usize = 5;
    const RUNS: usize = 50;

    println!(
        "Franka Panda bench: real URDF dynamics, nv={nv} (7 arm + {} finger), \
         decision vars: Explicit {}, AccelSpace {}, ForceSpace {}",
        nv - 7,
        nv + na,
        nv,
        na
    );
    println!("{RUNS} timed runs each (median), after {WARMUP} warm-ups.\n");

    let mut reference: Option<Extracted> = None;

    println!(
        "| {:<10} | {:<11} | {:<9} | {:>8} | {:>9} | {:>9} | {:>8} | {:>8} | {:>7} |",
        "form", "strategy", "backend", "med [ms]", "Δq̈", "Δτ", "eom", "ee-task", "τ_peak"
    );
    println!("|{}|", "-".repeat(104));

    for formulation in formulations {
        let d = Dynamics::new(formulation, &mass, &h, &jc, na);
        let levels = stack(&d);
        for (strategy, sname) in strategies {
            for (backend, bname) in backends {
                let cfg = SolveConfig { strategy, backend, ..Default::default() };

                let mut times = Vec::with_capacity(RUNS);
                let mut last = None;
                let mut status_ok = true;
                for run in 0..(WARMUP + RUNS) {
                    let t0 = Instant::now();
                    let sol = solve(&levels, &cfg).expect("solve");
                    let dt = t0.elapsed().as_secs_f64() * 1e3;
                    if run >= WARMUP {
                        times.push(dt);
                    }
                    status_ok &= sol.status == SolveStatus::Optimal;
                    last = Some(sol);
                }
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = times[times.len() / 2];

                let e = d.extract(&last.unwrap().x);
                let (eom, ee, tau_peak) = gauges(&e);
                let (dq, dt_) = match &reference {
                    None => (0.0, 0.0),
                    Some(r) => ((&e.qddot - &r.qddot).norm(), (&e.tau - &r.tau).norm()),
                };
                if reference.is_none() {
                    reference = Some(e);
                }

                println!(
                    "| {:<10} | {:<11} | {:<9} | {:>8.3} | {:>9.2e} | {:>9.2e} | {:>8.1e} | {:>8.1e} | {:>7.2} |{}",
                    format!("{formulation:?}"),
                    sname,
                    bname,
                    med,
                    dq,
                    dt_,
                    eom,
                    ee,
                    tau_peak,
                    if status_ok { "" } else { "  DEGRADED" }
                );
            }
        }
    }

    println!(
        "\nΔ columns: distance to the reference (Explicit + NullSpace + Clarabel).\n\
         eom: ‖M·q̈ + h − τ‖ (fully actuated, S = I) — must be ~0 in every row.\n\
         ee-task: ‖J·q̈ + J̇v − a_ref‖ — how well the priority-1 objective was met."
    );
}
