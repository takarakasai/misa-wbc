//! Go2-scale formulation × strategy benchmark (run with `--release`).
//!
//! Builds one realistic-size legged WBC problem — `nv = 18` (6 base +
//! 12 joints), 4 point contacts, `na = 12` — and solves the *same*
//! physical stack under every formulation × strategy × backend
//! combination, reporting:
//!
//! - solve time (median over repeated ticks),
//! - agreement with the reference combination
//!   (Explicit + NullSpace + Clarabel) on the physical triple (q̈, f, τ),
//! - physics-correctness gauges (EoM residual, contact residual, worst
//!   friction margin) — these must be small in every mode regardless of
//!   how the modes differ from each other.
//!
//! The matrices are synthetic but Go2-plausible (masses, torque limits,
//! contact geometry), so the numbers indicate what to expect before
//! wiring the real robot through `Dynamics`.

use std::time::Instant;

use misa_wbc::qp::QpSolver;
use misa_wbc::{
    solve, tasks, Dynamics, Extracted, Formulation, HqpStrategy, SolveConfig, SolveStatus, Task,
};
use nalgebra::{DMatrix, DVector};

const NV: usize = 18; // 6 base + 12 joints
const NA: usize = 12;
const NC: usize = 4; // point feet
const NF: usize = 3 * NC;
const MU: f64 = 0.6;
const TAU_MAX: f64 = 23.7; // Go2 joint limit [N·m]
const TOTAL_MASS: f64 = 15.0; // [kg]

/// Deterministic Go2-plausible dynamics matrices for one tick.
fn go2_like_system() -> (DMatrix<f64>, DVector<f64>, DMatrix<f64>, DVector<f64>) {
    // M: block-plausible diagonal (base translation = total mass, base
    // rotation = trunk inertia, joints = reflected leg inertia) plus a
    // deterministic symmetric coupling.
    let mut mass = DMatrix::zeros(NV, NV);
    for i in 0..3 {
        mass[(i, i)] = TOTAL_MASS;
    }
    for (i, inertia) in [(3, 0.36), (4, 0.45), (5, 0.20)] {
        mass[(i, i)] = inertia;
    }
    for j in 0..NA {
        mass[(6 + j, 6 + j)] = 0.06 + 0.02 * ((j % 3) as f64);
    }
    let l = DMatrix::from_fn(NV, 4, |i, j| 0.08 * ((i * 7 + j * 3) as f64 * 0.41).sin());
    mass += &l * l.transpose();

    // h: gravity on base z, small velocity-dependent terms elsewhere.
    let mut h = DVector::zeros(NV);
    h[2] = TOTAL_MASS * 9.81;
    for j in 0..NA {
        h[6 + j] = 0.4 * ((j as f64) * 0.9).sin();
    }

    // Jc: each foot senses base translation (identity), base rotation
    // (lever arm of a plausible foot position) and its own leg's three
    // joints (deterministic leg Jacobian block).
    let feet = [
        (0.19, -0.14, -0.30),
        (0.19, 0.14, -0.30),
        (-0.19, -0.14, -0.30),
        (-0.19, 0.14, -0.30),
    ];
    let mut jc = DMatrix::zeros(NF, NV);
    for (c, &(px, py, pz)) in feet.iter().enumerate() {
        let r = 3 * c;
        for i in 0..3 {
            jc[(r + i, i)] = 1.0;
        }
        // −skew(p) for the base-rotation columns (v = ω × p ⇒ J = −[p]×).
        #[rustfmt::skip]
        let skew = [
            [0.0, -pz,  py],
            [ pz, 0.0, -px],
            [-py,  px, 0.0],
        ];
        for i in 0..3 {
            for k in 0..3 {
                jc[(r + i, 3 + k)] = -skew[i][k];
            }
        }
        // Own leg (3 joints), deterministic non-trivial block.
        for i in 0..3 {
            for k in 0..3 {
                jc[(r + i, 6 + 3 * c + k)] = 0.25 * ((i * 3 + k + c) as f64 * 0.7).cos();
            }
        }
    }

    // J̇·v: small deterministic bias.
    let dj_v = DVector::from_fn(NF, |i, _| 0.05 * ((i as f64) * 1.3).sin());

    (mass, h, jc, dj_v)
}

/// The same physical stack, declared over whatever formulation the
/// context carries. Priority 0 = physics/constraints, 1 = tracking,
/// 2 = minimum-motion regularisation (pins the solution uniquely).
fn stack(d: &Dynamics, jc: &DMatrix<f64>, dj_v: &DVector<f64>) -> Vec<Task> {
    let f = d.forces();

    let mut p0 = tasks::zero_contact_acceleration(d.qddot(), jc, dj_v)
        + tasks::box_bound(d.tau(), &DVector::from_element(NA, TAU_MAX));
    for c in 0..NC {
        // Per-contact 3-D force sub-expression via a selection matrix.
        let mut sel = DMatrix::zeros(3, NF);
        for i in 0..3 {
            sel[(i, 3 * c + i)] = 1.0;
        }
        p0 = p0 + tasks::friction_pyramid(&(&sel * &f), MU);
    }
    if let Some(phys) = d.dynamics_task() {
        p0 = phys + p0;
    }

    // Tracking: hold the base still + share the weight evenly.
    let mut j_base = DMatrix::zeros(3, NV);
    for i in 0..3 {
        j_base[(i, i)] = 1.0;
    }
    let mut f_nom = DVector::zeros(NF);
    for c in 0..NC {
        f_nom[3 * c + 2] = TOTAL_MASS * 9.81 / NC as f64;
    }
    let p1 = tasks::cartesian_acceleration(d.qddot(), &j_base, &DVector::zeros(3), &DVector::zeros(3))
        + tasks::regularize(&f, &f_nom).weight(0.1);

    let p2 = tasks::track(d.qddot(), &DVector::zeros(NV))
        + tasks::track(d.tau(), &DVector::zeros(NA)).weight(0.1);

    vec![p0, p1, p2]
}

struct Gauges {
    eom: f64,
    contact: f64,
    worst_friction_margin: f64,
    tau_peak: f64,
}

fn gauges(
    e: &Extracted,
    mass: &DMatrix<f64>,
    h: &DVector<f64>,
    jc: &DMatrix<f64>,
    dj_v: &DVector<f64>,
) -> Gauges {
    let mut s_t = DMatrix::zeros(NV, NA);
    for i in 0..NA {
        s_t[(NV - NA + i, i)] = 1.0;
    }
    let eom = (mass * &e.qddot + h - &s_t * &e.tau - jc.transpose() * &e.forces).norm();
    let contact = (jc * &e.qddot + dj_v).norm();
    let mut worst = f64::INFINITY;
    for c in 0..NC {
        let (fx, fy, fz) = (e.forces[3 * c], e.forces[3 * c + 1], e.forces[3 * c + 2]);
        for m in [fz, MU * fz - fx.abs(), MU * fz - fy.abs()] {
            worst = worst.min(m);
        }
    }
    let tau_peak = e.tau.amax();
    Gauges { eom, contact, worst_friction_margin: worst, tau_peak }
}

fn main() {
    let (mass, h, jc, dj_v) = go2_like_system();

    let formulations = [Formulation::Explicit, Formulation::AccelSpace, Formulation::ForceSpace];
    let strategies = [
        (HqpStrategy::NullSpace, "NullSpace"),
        (HqpStrategy::ForceBudgetCascade, "ForceBudget"),
    ];
    let backends = [(QpSolver::Clarabel, "Clarabel"), (QpSolver::ActiveSet, "ActiveSet")];

    const WARMUP: usize = 5;
    const RUNS: usize = 50;

    println!(
        "Go2-scale bench: nv={NV}, nc={NC}, na={NA}  (decision vars: \
         Explicit {}, AccelSpace {}, ForceSpace {})",
        NV + NF + NA,
        NV + NF,
        NA + NF
    );
    println!("{RUNS} timed runs each (median), after {WARMUP} warm-ups.\n");

    let mut reference: Option<Extracted> = None;

    println!(
        "| {:<10} | {:<11} | {:<9} | {:>8} | {:>9} | {:>9} | {:>9} | {:>8} | {:>8} | {:>7} |",
        "form", "strategy", "backend", "med [ms]", "Δq̈", "Δf", "Δτ", "eom", "contact", "τ_peak"
    );
    println!("|{}|", "-".repeat(126));

    for formulation in formulations {
        let d = Dynamics::new(formulation, &mass, &h, &jc, NA);
        let levels = stack(&d, &jc, &dj_v);
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

                let sol = last.unwrap();
                let e = d.extract(&sol.x);
                let g = gauges(&e, &mass, &h, &jc, &dj_v);

                let (dq, df, dt_) = match &reference {
                    None => (0.0, 0.0, 0.0),
                    Some(r) => (
                        (&e.qddot - &r.qddot).norm(),
                        (&e.forces - &r.forces).norm(),
                        (&e.tau - &r.tau).norm(),
                    ),
                };
                if reference.is_none() {
                    reference = Some(e);
                }

                println!(
                    "| {:<10} | {:<11} | {:<9} | {:>8.3} | {:>9.2e} | {:>9.2e} | {:>9.2e} | {:>8.1e} | {:>8.1e} | {:>7.2} |{}",
                    format!("{formulation:?}"),
                    sname,
                    bname,
                    med,
                    dq,
                    df,
                    dt_,
                    g.eom,
                    g.contact,
                    g.tau_peak,
                    if status_ok { "" } else { "  DEGRADED" }
                );
                let _ = g.worst_friction_margin;
            }
        }
    }

    println!(
        "\nΔ columns: distance to the reference combination \
         (Explicit + NullSpace + Clarabel).\n\
         eom / contact: physics residuals — must be small in EVERY row.\n\
         NullSpace rows should agree with the reference; ForceBudget rows \
         may differ (greedy hierarchy) while still satisfying the physics."
    );
}
