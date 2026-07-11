//! Warm-started active set vs cold vs Clarabel, on a Go2-scale QP tick
//! sequence (run with `--release`).
//!
//! The scenario is the qpOASES use-case: a whole-body controller solves
//! an almost-identical QP every tick. Here the QP is the Go2-scale
//! stack of `formulation_bench` collapsed into one weighted problem
//! (n = 42, 44 inequality rows), with the tracking references drifting
//! sinusoidally across 100 ticks. Three ways to solve the sequence:
//!
//! - **ActiveSet cold** — workspace cleared every tick (the old
//!   behaviour, plus the incremental factor updates).
//! - **ActiveSet warm** — one persistent [`QpWorkspace`]: each tick
//!   starts from the previous optimum and working set.
//! - **Clarabel** — the interior-point baseline (no warm start).
//!
//! Reported per solver: median / worst tick time, total iterations
//! (active set only), and cross-checked solutions.

use std::time::Instant;

use misa_wbc::qp::{solve_qp, solve_qp_warm, QpConfig, QpSolver, QpStatus, QpWorkspace};
use misa_wbc::{tasks, Dynamics, Formulation, Task};
use nalgebra::{DMatrix, DVector};

const NV: usize = 18;
const NA: usize = 12;
const NC: usize = 4;
const NF: usize = 3 * NC;
const MU: f64 = 0.6;
const TAU_MAX: f64 = 23.7;
const TOTAL_MASS: f64 = 15.0;
const TICKS: usize = 100;

/// Same deterministic Go2-plausible system as `formulation_bench`.
fn go2_like_system() -> (DMatrix<f64>, DVector<f64>, DMatrix<f64>, DVector<f64>) {
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

    let mut h = DVector::zeros(NV);
    h[2] = TOTAL_MASS * 9.81;
    for j in 0..NA {
        h[6 + j] = 0.4 * ((j as f64) * 0.9).sin();
    }

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
        for i in 0..3 {
            for k in 0..3 {
                jc[(r + i, 6 + 3 * c + k)] = 0.25 * ((i * 3 + k + c) as f64 * 0.7).cos();
            }
        }
    }

    let dj_v = DVector::from_fn(NF, |i, _| 0.05 * ((i as f64) * 1.3).sin());
    (mass, h, jc, dj_v)
}

/// The stack collapsed to one weighted QP: H = Σ wₗ AₗᵀAₗ + εI,
/// c(t) = −Σ wₗ Aₗᵀ bₗ(t), D x ≤ f. The tick parameter shifts the
/// tracking references (base target + force distribution), which is
/// exactly what changes between control ticks.
struct TickProblem {
    h: DMatrix<f64>,
    d: DMatrix<f64>,
    f: DVector<f64>,
    // per-level (weight, A, base-b) so c(t) can be re-derived
    parts: Vec<(f64, DMatrix<f64>, DVector<f64>)>,
}

fn build_problem() -> TickProblem {
    let (mass, h_dyn, jc, dj_v) = go2_like_system();
    let d = Dynamics::new(Formulation::Explicit, &mass, &h_dyn, &jc, NA);
    let f = d.forces();

    let mut p0 = tasks::zero_contact_acceleration(d.qddot(), &jc, &dj_v)
        + tasks::box_bound(d.tau(), &DVector::from_element(NA, TAU_MAX));
    for c in 0..NC {
        let mut sel = DMatrix::zeros(3, NF);
        for i in 0..3 {
            sel[(i, 3 * c + i)] = 1.0;
        }
        p0 = p0 + tasks::friction_pyramid(&(&sel * &f), MU);
    }
    let p0 = d.dynamics_task().unwrap() + p0;

    let mut j_base = DMatrix::zeros(3, NV);
    for i in 0..3 {
        j_base[(i, i)] = 1.0;
    }
    let mut f_nom = DVector::zeros(NF);
    for c in 0..NC {
        f_nom[3 * c + 2] = TOTAL_MASS * 9.81 / NC as f64;
    }
    let p1 = tasks::cartesian_acceleration(d.qddot(), &j_base, &DVector::zeros(3), &DVector::zeros(3))
        + tasks::regularize(&f, &f_nom);
    let n = d.layout().n_decision();
    let p2 = tasks::track(d.qddot(), &DVector::zeros(NV))
        + tasks::track(d.tau(), &DVector::zeros(NA));

    let weights = [1.0, 0.1, 1e-3];
    let levels: Vec<Task> = vec![p0, p1, p2];

    let mut h = DMatrix::<f64>::identity(n, n) * 1e-8;
    let mut d_rows: Vec<(DMatrix<f64>, DVector<f64>)> = Vec::new();
    let mut parts = Vec::new();
    for (t, &w) in levels.iter().zip(weights.iter()) {
        if t.n_eq() > 0 {
            h += w * t.a.transpose() * &t.a;
            parts.push((w, t.a.clone(), t.b.clone()));
        }
        if t.n_iq() > 0 {
            d_rows.push((t.d.clone(), t.f.clone()));
        }
    }
    let m_iq: usize = d_rows.iter().map(|(m, _)| m.nrows()).sum();
    let mut dmat = DMatrix::zeros(m_iq, n);
    let mut fvec = DVector::zeros(m_iq);
    let mut r = 0;
    for (m, fv) in d_rows {
        dmat.rows_mut(r, m.nrows()).copy_from(&m);
        fvec.rows_mut(r, fv.len()).copy_from(&fv);
        r += m.nrows();
    }
    TickProblem { h, d: dmat, f: fvec, parts }
}

/// c(t): the tracking b's drift sinusoidally (base sway + force shift).
fn cost_at_tick(p: &TickProblem, tick: usize) -> DVector<f64> {
    let n = p.h.nrows();
    let phase = tick as f64 * 0.05; // gentle tick-to-tick change
    let mut c = DVector::zeros(n);
    for (li, (w, a, b0)) in p.parts.iter().enumerate() {
        let b = DVector::from_fn(b0.len(), |i, _| {
            b0[i] + 0.02 * (phase + (i + li * 7) as f64 * 0.6).sin()
        });
        c -= *w * a.transpose() * b;
    }
    c
}

struct Run {
    name: &'static str,
    med_ms: f64,
    worst_ms: f64,
    iters: usize,
    degraded: usize,
    x_last: DVector<f64>,
}

fn main() {
    let p = build_problem();
    let n = p.h.nrows();
    println!(
        "Warm-start bench: one weighted Go2-scale QP (n = {n}, {} iq rows), {TICKS} drifting ticks.\n",
        p.d.nrows()
    );

    let as_cfg = QpConfig { solver: QpSolver::ActiveSet, max_iters: 2000, ..Default::default() };
    let cl_cfg = QpConfig { solver: QpSolver::Clarabel, ..Default::default() };

    let mut runs: Vec<Run> = Vec::new();

    let ipm_cfg = QpConfig { solver: QpSolver::Ipm, ..Default::default() };
    let admm_cfg = QpConfig { solver: QpSolver::Admm, ..Default::default() };

    for mode in ["ActiveSet cold", "ActiveSet warm", "Ipm", "Admm", "Clarabel"] {
        let mut ws = QpWorkspace::new();
        let mut times = Vec::with_capacity(TICKS);
        let mut iters = 0usize;
        let mut degraded = 0usize;
        let mut x_last = DVector::zeros(n);

        for tick in 0..TICKS {
            let c = cost_at_tick(&p, tick);
            let t0 = Instant::now();
            let sol = match mode {
                "ActiveSet cold" => {
                    ws.clear();
                    solve_qp_warm(&p.h, &c, None, None, Some(&p.d), Some(&p.f), None, &as_cfg, &mut ws)
                }
                "ActiveSet warm" => {
                    solve_qp_warm(&p.h, &c, None, None, Some(&p.d), Some(&p.f), None, &as_cfg, &mut ws)
                }
                "Ipm" => solve_qp(&p.h, &c, None, None, Some(&p.d), Some(&p.f), None, &ipm_cfg),
                "Admm" => solve_qp(&p.h, &c, None, None, Some(&p.d), Some(&p.f), None, &admm_cfg),
                _ => solve_qp(&p.h, &c, None, None, Some(&p.d), Some(&p.f), None, &cl_cfg),
            };
            times.push(t0.elapsed().as_secs_f64() * 1e3);
            iters += sol.iterations;
            if sol.status != QpStatus::Optimal {
                degraded += 1;
            }
            x_last = sol.x;
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        runs.push(Run {
            name: mode,
            med_ms: times[TICKS / 2],
            worst_ms: times[TICKS - 1],
            iters,
            degraded,
            x_last,
        });
    }

    println!(
        "| {:<15} | {:>9} | {:>9} | {:>11} | {:>8} |",
        "solver", "med [ms]", "max [ms]", "total iters", "degraded"
    );
    println!("|{}|", "-".repeat(68));
    for r in &runs {
        println!(
            "| {:<15} | {:>9.3} | {:>9.3} | {:>11} | {:>8} |",
            r.name,
            r.med_ms,
            r.worst_ms,
            if r.iters > 0 { r.iters.to_string() } else { "—".into() },
            r.degraded
        );
    }

    // Cross-check: all three land on the same final-tick solution.
    let ref_x = &runs[2].x_last;
    for r in &runs[..2] {
        println!(
            "‖x_{} − x_Clarabel‖ = {:.2e}",
            r.name.replace("ActiveSet ", ""),
            (&r.x_last - ref_x).norm()
        );
    }
}
