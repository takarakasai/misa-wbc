//! Singularity-approach stability comparison: the Panda's end-effector
//! is driven back and forth along a fixed radial direction from the
//! shoulder, out to near the edge of reach in that direction — the
//! generic "arm straightens out" boundary singularity, where
//! `sigma_min(J_lin)` collapses toward zero regardless of which joints
//! happen to carry the motion.
//!
//! Same task design as `panda_circle_demo.rs` (dynamics_task +
//! box_bound(tau) at priority 0, cartesian_acceleration at priority 1,
//! regularization at priority 2), parameterized over `Formulation` and
//! `QpSolver` so the same trajectory can be re-run under each
//! combination for a controlled comparison:
//!
//! ```text
//! cargo run --release --example panda_singularity_demo -- <formulation> <backend> [damped]
//! formulation: explicit | accel | force
//! backend:     activeset | ipm | admm | clarabel
//! damped:      literal "damped" to swap in tasks::cartesian_acceleration_damped
//!              (SingularityDamping::default(), or override lambda_max_sq via the
//!              LAMBDA_MAX_SQ env var) -- see ref/wbc_comparison.md Sec.5n for the
//!              backend-dependent results (reliable win on Ipm/Admm, validate
//!              first on ActiveSet/Clarabel -- one measured combination diverges).
//! ```
//!
//! CSV column layout is a superset of `panda_circle_demo.rs`'s (same
//! prefix: tick,t,q0..8,ee_xyz,ref_xyz,link_poses...), with six
//! diagnostic columns appended at the end so the existing
//! `render_panda_vtk.py` reads this file unchanged:
//! `sigma_min,tau_norm,tau_max_abs,qddot_norm,status_code,degraded_level`.

use misarta::fk::forward_kinematics;
use misarta::jacobian::{compute_jacobian_dot_times_v, compute_joint_jacobian};
use misarta::se3;

use misa_wbc::{refgen, tasks, Dynamics, Formulation, QpSolver, SolveConfig, Solver};
use nalgebra::{DMatrix, DVector, Vector3};

const DT: f64 = 0.004; // 250 Hz
const DURATION_S: f64 = 5.0;
const REACH_CENTER: f64 = 0.68;
const REACH_AMP: f64 = 0.13; // sweeps reach in [0.55, 0.81] m from the shoulder
const PERIOD_S: f64 = 2.5; // two full in/out cycles over DURATION_S

fn parse_formulation(s: &str) -> Formulation {
    match s {
        "explicit" => Formulation::Explicit,
        "accel" => Formulation::AccelSpace,
        "force" => Formulation::ForceSpace,
        other => panic!("unknown formulation '{other}' (expected explicit|accel|force)"),
    }
}

fn parse_backend(s: &str) -> QpSolver {
    match s {
        "activeset" => QpSolver::ActiveSet,
        "ipm" => QpSolver::Ipm,
        "admm" => QpSolver::Admm,
        "clarabel" => QpSolver::Clarabel,
        other => panic!("unknown backend '{other}' (expected activeset|ipm|admm|clarabel)"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let formulation = parse_formulation(args.get(1).map(|s| s.as_str()).unwrap_or("explicit"));
    let backend = parse_backend(args.get(2).map(|s| s.as_str()).unwrap_or("activeset"));
    let damped = args.get(3).map(|s| s.as_str()) == Some("damped");
    eprintln!("formulation={:?} backend={:?} damped={damped}", formulation, backend);

    let path = format!("{}/examples/models/panda.urdf", env!("CARGO_MANIFEST_DIR"));
    let imported = misarta_formats::urdf::import(std::path::Path::new(&path)).expect("import panda.urdf");
    let (model, _vis, _col) = misarta::native::build_model(&imported.file).expect("build panda model");
    let (nv, nq) = (model.nv, model.nq);
    assert_eq!(nq, nv, "fixed-base arm expected");
    let na = nv;

    let ee_idx = model
        .joints
        .iter()
        .position(|j| j.name == "panda_joint7")
        .expect("panda_joint7 in model");

    // Ready pose, at rest -- same as panda_circle_demo.
    let mut q = vec![0.0_f64; nq];
    for (i, &qi) in [0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.8].iter().enumerate() {
        q[i] = qi;
    }
    let mut v = vec![0.0_f64; nv];

    let mut tau_max = DVector::from_element(na, 20.0);
    for (i, &t) in [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0].iter().enumerate() {
        tau_max[i] = t;
    }
    let kp = DVector::from_element(6, 400.0);
    let kd = DVector::from_element(6, 40.0);

    // Fixed radial direction from the shoulder (panda_joint1's origin,
    // (0,0,0.333) for every q -- it only rotates about Z there) through
    // the resting EE position. Reaching along this direction toward its
    // far end straightens the arm, collapsing sigma_min(J_lin) -- the
    // generic reach-boundary singularity, independent of which joints
    // end up carrying the motion.
    let shoulder = Vector3::new(0.0, 0.0, 0.333);
    let data0 = forward_kinematics(&model, &q);
    let ee0 = se3::translation(&data0.oMi[ee_idx]);
    let dir = (ee0 - shoulder).normalize();

    let cfg = SolveConfig { backend, ..SolveConfig::default() }; // strategy: NullSpace (default)
    let mut solver = Solver::new();

    let n_ticks = (DURATION_S / DT) as usize;
    for tick in 0..n_ticks {
        let t = tick as f64 * DT;
        let omega = 2.0 * std::f64::consts::PI / PERIOD_S;
        let phase = omega * t;

        let reach = REACH_CENTER + REACH_AMP * phase.cos();
        let reach_dot = -REACH_AMP * omega * phase.sin();
        let x_ref = shoulder + reach * dir;
        let v_ref = reach_dot * dir;

        let mass = misarta::crba::crba(&model, &q);
        let h = misarta::rnea::nonlinear_effects(&model, &q, &v);
        let j_ee = compute_joint_jacobian(&model, &q, ee_idx);
        let dj_v6 = compute_jacobian_dot_times_v(&model, &q, &v, ee_idx);
        let dj_v = DVector::from_column_slice(dj_v6.as_slice());

        // Manipulability proxy: smallest singular value of the linear
        // (bottom 3) rows of J_ee -- how close the current pose is to
        // the reach-boundary singularity, continuously.
        let j_lin = j_ee.rows(3, 3).into_owned();
        let svd = nalgebra::linalg::SVD::new(j_lin, false, false);
        let sigma_min = svd.singular_values.iter().cloned().fold(f64::INFINITY, f64::min);

        let data = forward_kinematics(&model, &q);
        let ee_pos = se3::translation(&data.oMi[ee_idx]);
        let ee_lin_vel = {
            let jv = &j_ee * DVector::from_column_slice(&v);
            Vector3::new(jv[3], jv[4], jv[5])
        };

        let a_lin = refgen::pd(
            &DVector::from_column_slice(x_ref.as_slice()),
            &DVector::from_column_slice(ee_pos.as_slice()),
            &DVector::from_column_slice(v_ref.as_slice()),
            &DVector::from_column_slice(ee_lin_vel.as_slice()),
            &kp.rows(3, 3).into_owned(),
            &kd.rows(3, 3).into_owned(),
        );
        let mut a_ref = DVector::zeros(6);
        a_ref.rows_mut(3, 3).copy_from(&a_lin);

        let d = Dynamics::new(formulation, &mass, &h, &DMatrix::zeros(0, nv), na);
        let mut p0 = tasks::box_bound(d.tau(), &tau_max);
        if let Some(phys) = d.dynamics_task() {
            p0 = phys + p0;
        }
        let p1 = if damped {
            let mut cfg = tasks::SingularityDamping::default();
            if let Ok(l) = std::env::var("LAMBDA_MAX_SQ") {
                cfg.lambda_max_sq = l.parse().expect("LAMBDA_MAX_SQ must be a float");
            }
            tasks::cartesian_acceleration_damped(d.qddot(), &j_ee, &dj_v, &a_ref, &cfg)
        } else {
            tasks::cartesian_acceleration(d.qddot(), &j_ee, &dj_v, &a_ref)
        };
        let p2 = tasks::track(d.qddot(), &DVector::zeros(nv))
            + tasks::track(d.tau(), &DVector::zeros(na)).weight(0.01);

        let sol = solver.solve(&[p0, p1, p2], &cfg).expect("solve");
        let e = d.extract(&sol.x);

        let (status_code, degraded_level) = match &sol.status {
            misa_wbc::SolveStatus::Optimal => (0, -1i64),
            misa_wbc::SolveStatus::Degraded { level, .. } => (1, *level as i64),
        };
        let tau_norm = e.tau.norm();
        let tau_max_abs = e.tau.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
        let qddot_norm = e.qddot.norm();

        for i in 0..nv {
            v[i] += e.qddot[i] * DT;
            q[i] += v[i] * DT;
        }

        // Numerical-divergence guard: a solver can keep reporting
        // SolveStatus::Optimal every tick while the *state* it's
        // integrating quietly runs away (measured on Explicit+Clarabel
        // under cartesian_acceleration_damped -- see
        // ref/wbc_comparison.md Sec.5n). Catch it here rather than
        // grinding through the rest of the simulation on garbage state
        // (which is also why that run took minutes instead of seconds).
        // 1e3 rad / rad-s^-1 is already ~300x past any real joint limit
        // or plausible velocity for this arm -- only genuine blow-up
        // trips it.
        if q.iter().chain(v.iter()).any(|x| !x.is_finite() || x.abs() > 1e3) {
            eprintln!("STATE DIVERGED at tick {tick} (t={t:.4}s) -- stopping early, trace truncated here");
            break;
        }

        print!("{tick},{t:.4}");
        for &qi in &q {
            print!(",{qi:.5}");
        }
        print!(",{:.5},{:.5},{:.5}", ee_pos.x, ee_pos.y, ee_pos.z);
        print!(",{:.5},{:.5},{:.5}", x_ref.x, x_ref.y, x_ref.z);
        for j in 0..model.joints.len() {
            let p = se3::translation(&data.oMi[j]);
            let r = se3::rotation_matrix(&data.oMi[j]);
            print!(",{:.5},{:.5},{:.5}", p.x, p.y, p.z);
            for row in 0..3 {
                for col in 0..3 {
                    print!(",{:.6}", r[(row, col)]);
                }
            }
        }
        print!(",{sigma_min:.6},{tau_norm:.5},{tau_max_abs:.5},{qddot_norm:.5},{status_code},{degraded_level}");
        println!();
    }
}
